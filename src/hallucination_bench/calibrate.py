"""Calibrate the taxonomy classifier on hand-labelled cases and apply the held-out gate.

Cases (``data/calibration/{dev,heldout}.jsonl``, schema in ``data/calibration/README.md``) carry
exactly what the classifier sees in production: a prompt, a marked response, readouts and a
span list with grounding; plus gold labels. Each case is classified ``repeats`` times (default
2). The repeat index (``r0``/``r1``) is part of the cache key and the fingerprint, so every
repeat is a separate API call. Gate metrics are the mean over repeats, with a Wilson 95% CI.

On ``dev`` every disagreement is reported with the classifier's ``why``; on ``heldout`` only
aggregates are reported (no per-span lists), so held-out errors cannot leak into prompt tuning.
"""

import asyncio
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hallucination_bench import DATA
from hallucination_bench import taxonomy as tx
from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.judges import taxonomy_v1 as tv1
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.llm import LLM, Usage

CAL_DIR = DATA / "calibration"
SPLITS = ("dev", "heldout")

# (measure, metric key, threshold, higher_is_better)
GATE: tuple[tuple[str, str, float, bool], ...] = (
    ("type accuracy", "type_accuracy", 0.85, True),
    ("concept recall", "concept_recall", 0.80, True),
    ("detail recall", "detail_recall", 0.80, True),
    ("severity exact", "severity_exact", 0.65, True),
    ("severity quadratic-weighted kappa", "severity_qwk", 0.60, True),
    ("target accuracy", "target_accuracy", 0.70, True),
    ("unsupported_kind accuracy (unsupported spans)", "unsupported_kind_accuracy", 0.70, True),
    ("n_unclassified", "n_unclassified", 0, False),
    ("type flip rate between runs", "type_flip_rate", 0.05, False),
    ("type accuracy, real part", "real_type_accuracy", 0.80, True),
)


def load_cases(split: str, path: Path | None = None) -> list[dict[str, Any]]:
    p = path or CAL_DIR / f"{split}.jsonl"
    cases = [
        json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    errs = [e for c in cases for e in validate_case(c)]
    if errs:
        raise ValueError(f"{p}: invalid cases:\n  " + "\n  ".join(errs[:30]))
    ids = Counter(c["case_id"] for c in cases)
    dup = [k for k, n in ids.items() if n > 1]
    if dup:
        raise ValueError(f"{p}: duplicate case_id {dup}")
    return cases


def validate_case(c: dict[str, Any], *, require_gold: bool = True) -> list[str]:
    """Schema errors for one case (empty when valid)."""
    cid = c.get("case_id", "?")
    errs: list[str] = []
    for k in ("case_id", "prompt", "response", "readouts", "spans"):
        if k not in c:
            errs.append(f"{cid}: missing {k}")
    if errs:
        return errs
    if c["response"].count(READ_MARK) != 1:
        errs.append(f"{cid}: response must contain {READ_MARK} exactly once")
    if not isinstance(c["readouts"], list) or not 1 <= len(c["readouts"]) <= 3:
        errs.append(f"{cid}: readouts must be a list of 1-3 strings")
        return errs
    if c.get("part", "synthetic") not in ("synthetic", "real"):
        errs.append(f"{cid}: part must be synthetic|real")
    if not c["spans"]:
        errs.append(f"{cid}: no spans")
    for j, s in enumerate(c["spans"]):
        tag = f"{cid}/span{j}"
        ri = s.get("readout_idx")
        if not isinstance(ri, int) or not 0 <= ri < len(c["readouts"]):
            errs.append(f"{tag}: bad readout_idx")
            continue
        if not s.get("text") or s["text"] not in c["readouts"][ri]:
            errs.append(f"{tag}: text is not a verbatim substring of readout {ri}")
        g = s.get("grounding")
        if g not in tv1.GROUNDING_ORDER:
            errs.append(f"{tag}: bad grounding {g!r}")
        gold = s.get("gold")
        if gold is None:
            if require_gold:
                errs.append(f"{tag}: missing gold")
            continue
        if gold.get("type") not in tx.TYPES:
            errs.append(f"{tag}: bad gold type")
        if gold.get("target") not in tx.TARGETS:
            errs.append(f"{tag}: bad gold target")
        if gold.get("severity") not in tx.SEVERITIES:
            errs.append(f"{tag}: bad gold severity")
        uk = gold.get("unsupported_kind")
        if g == "unsupported" and uk not in tx.UNSUPPORTED_KINDS:
            errs.append(f"{tag}: unsupported span needs a gold unsupported_kind")
        if g != "unsupported" and uk is not None:
            errs.append(f"{tag}: unsupported_kind must be null for {g}")
    return errs


def case_spans(c: dict[str, Any]) -> list[tv1.SpanIn]:
    return [tv1.SpanIn(s["readout_idx"], s["text"], s["grounding"]) for s in c["spans"]]


def case_prompt(c: dict[str, Any]) -> tuple[str, str]:
    readouts = list(enumerate(c["readouts"]))
    return tv1.render_prompt(c["prompt"], c["response"], readouts, case_spans(c))


def cal_key(split: str, case_id: str, r: int) -> str:
    return f"{split}:{case_id}:r{r}"


async def classify_cases(
    cases: Sequence[dict[str, Any]],
    *,
    split: str,
    model: str,
    effort: str | None,
    cache: Cache,
    llm: LLM | None,
    repeats: int = 2,
    usage: Usage | None = None,
) -> list[list[list[tv1.SpanLabel]]]:
    """labels[r][case_i] = the parsed labels of repeat ``r`` (fail closed)."""
    usage = usage if usage is not None else Usage()
    out: list[list[list[tv1.SpanLabel]]] = [[[] for _ in cases] for _ in range(repeats)]

    async def one(r: int, i: int, c: dict[str, Any]) -> None:
        system, user = case_prompt(c)
        spans = case_spans(c)
        key = cal_key(split, c["case_id"], r)
        fp = fingerprint(tv1.PROMPT_VERSION, model, effort, f"r{r}", system, user)
        row = cache.get(key, fp)
        raw: dict[str, Any] | None
        if row is not None:
            raw = row["result"]
        elif llm is None:
            raw = None
        else:
            res = await llm.call(model=model, system=system, user=user, schema=tv1.SCHEMA)
            usage.add(res.usage)
            meta = {"stage": "calibrate", "usage": res.usage.to_dict()}
            raw = res.result
            if raw is not None and not tv1.validate(raw, spans):
                cache.put(key, fp, {"result": None, "meta": {**meta, "raw": raw}})
            else:
                cache.put(key, fp, {"result": raw, "meta": meta})
        out[r][i], _ = tv1.parse(raw, spans)

    if llm is not None:
        await llm.preflight(model)
    await asyncio.gather(*(one(r, i, c) for r in range(repeats) for i, c in enumerate(cases)))
    return out


# ---------------------------------------------------------------- metrics


def wilson(p: float | None, n: int, z: float = 1.959964) -> list[float] | None:
    if p is None or n == 0:
        return None
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def qwk(gold: Sequence[int], pred: Sequence[int], cats: Sequence[int] = (1, 2, 3)) -> float | None:
    """Quadratic-weighted Cohen's kappa."""
    if not gold:
        return None
    k = len(cats)
    ix = {c: i for i, c in enumerate(cats)}
    obs = [[0.0] * k for _ in range(k)]
    for g, p in zip(gold, pred, strict=True):
        obs[ix[g]][ix[p]] += 1
    n = len(gold)
    rg = [sum(row) for row in obs]
    cp = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * rg[i] * cp[j] / n
    return 1.0 - num / den if den else None


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def run_metrics(cases: Sequence[dict[str, Any]], labels: Sequence[list[tv1.SpanLabel]]) -> dict:
    """Metrics of one repeat."""
    c: Counter = Counter()
    sev_g: list[int] = []
    sev_p: list[int] = []
    for case, labs in zip(cases, labels, strict=True):
        real = case.get("part") == "real"
        for s, lab in zip(case["spans"], labs, strict=True):
            g = s["gold"]
            c["n"] += 1
            c["type_ok"] += lab.type == g["type"]
            c["n_unclassified"] += lab.type == tx.UNKNOWN
            c[f"gold_{g['type']}"] += 1
            c[f"hit_{g['type']}"] += lab.type == g["type"]
            c["target_ok"] += lab.target == g["target"]
            c["sev_exact"] += lab.severity == g["severity"]
            c["sev_within1"] += lab.severity is not None and abs(lab.severity - g["severity"]) <= 1
            if lab.severity is not None:
                sev_g.append(g["severity"])
                sev_p.append(lab.severity)
            if s["grounding"] == "unsupported":
                c["n_unsup"] += 1
                c["uk_ok"] += lab.unsupported_kind == g["unsupported_kind"]
            if real:
                c["n_real"] += 1
                c["real_type_ok"] += lab.type == g["type"]
    return {
        "n_spans": c["n"],
        "type_accuracy": _ratio(c["type_ok"], c["n"]),
        "concept_recall": _ratio(c["hit_concept"], c["gold_concept"]),
        "detail_recall": _ratio(c["hit_detail"], c["gold_detail"]),
        "target_accuracy": _ratio(c["target_ok"], c["n"]),
        "severity_exact": _ratio(c["sev_exact"], c["n"]),
        "severity_within1": _ratio(c["sev_within1"], c["n"]),
        "severity_qwk": qwk(sev_g, sev_p),
        "unsupported_kind_accuracy": _ratio(c["uk_ok"], c["n_unsup"]),
        "real_type_accuracy": _ratio(c["real_type_ok"], c["n_real"]),
        "n_unclassified": c["n_unclassified"],
        "_n": {
            "type_accuracy": c["n"],
            "concept_recall": c["gold_concept"],
            "detail_recall": c["gold_detail"],
            "target_accuracy": c["n"],
            "severity_exact": c["n"],
            "severity_within1": c["n"],
            "unsupported_kind_accuracy": c["n_unsup"],
            "real_type_accuracy": c["n_real"],
        },
    }


def _mean(xs: Sequence[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def confusion(
    cases: Sequence[dict[str, Any]], labels: Sequence[Sequence[list[tv1.SpanLabel]]], field: str
) -> dict[str, dict[str, int]]:
    """gold -> pred -> count, pooled over repeats."""
    m: dict[str, Counter] = {}
    for rep in labels:
        for case, labs in zip(cases, rep, strict=True):
            for s, lab in zip(case["spans"], labs, strict=True):
                if field == "unsupported_kind" and s["grounding"] != "unsupported":
                    continue
                g = str(s["gold"][field])
                m.setdefault(g, Counter())[str(getattr(lab, field))] += 1
    return {g: dict(sorted(c.items())) for g, c in sorted(m.items())}


def score(
    cases: Sequence[dict[str, Any]], labels: Sequence[Sequence[list[tv1.SpanLabel]]], split: str
) -> dict[str, Any]:
    runs = [run_metrics(cases, rep) for rep in labels]
    keys = [k for k in runs[0] if not k.startswith("_") and k != "n_spans"]
    mean = {k: _mean([r[k] for r in runs]) for k in keys}
    mean["n_unclassified"] = max(r["n_unclassified"] for r in runs)
    flips: dict[str, float | None] = {}
    if len(labels) >= 2:
        n = 0
        diff: Counter = Counter()
        for case_i in range(len(cases)):
            for a, b in zip(labels[0][case_i], labels[1][case_i], strict=True):
                n += 1
                for f in ("type", "target", "severity", "unsupported_kind"):
                    diff[f] += getattr(a, f) != getattr(b, f)
        flips = {
            f"{f}_flip_rate": _ratio(diff[f], n)
            for f in ("type", "target", "severity", "unsupported_kind")
        }
    mean.update(flips)
    ns = runs[0]["_n"] | {"type_flip_rate": runs[0]["n_spans"]}
    ci = {k: wilson(mean.get(k), ns[k]) for k in ns if mean.get(k) is not None}
    out: dict[str, Any] = {
        "split": split,
        "prompt_version": tv1.PROMPT_VERSION,
        "n_cases": len(cases),
        "n_spans": runs[0]["n_spans"],
        "n": ns,
        "mean": mean,
        "wilson95": ci,
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
        "confusion": {
            f: confusion(cases, labels, f)
            for f in ("type", "target", "severity", "unsupported_kind")
        },
        "gate": gate(mean, split),
    }
    out["gate_passed"] = all(g["passed"] is not False for g in out["gate"])
    if split == "dev":
        out["disagreements"] = disagreements(cases, labels)
    return out


def gate(mean: dict[str, Any], split: str = "heldout") -> list[dict[str, Any]]:
    """The gate rows. The gate applies to ``heldout``; on ``dev`` a metric with no data (the
    real-part accuracy: dev is synthetic only) is ``n/a`` (``passed = None``), not a fail."""
    rows = []
    for name, key, thr, hib in GATE:
        v = mean.get(key)
        ok: bool | None = v is not None and (v >= thr if hib else v <= thr)
        if v is None and split != "heldout":
            ok = None
        rows.append({"measure": name, "key": key, "value": v, "threshold": thr, "passed": ok})
    return rows


def disagreements(
    cases: Sequence[dict[str, Any]], labels: Sequence[Sequence[list[tv1.SpanLabel]]]
) -> list[dict[str, Any]]:
    out = []
    for r, rep in enumerate(labels):
        for case, labs in zip(cases, rep, strict=True):
            for s, lab in zip(case["spans"], labs, strict=True):
                g = s["gold"]
                bad = {
                    f: {"gold": g[f], "pred": getattr(lab, f)}
                    for f in ("type", "target", "severity", "unsupported_kind")
                    if (f != "unsupported_kind" or s["grounding"] == "unsupported")
                    and getattr(lab, f) != g[f]
                }
                if bad:
                    out.append(
                        {
                            "run": r,
                            "case_id": case["case_id"],
                            "rule": case.get("rule"),
                            "span_id": lab.span_id,
                            "grounding": s["grounding"],
                            "text": s["text"],
                            "diff": bad,
                            "why": lab.why,
                        }
                    )
    return out


def render_markdown(res: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    lines = [
        f"## Calibration: {res['split']} ({res['prompt_version']}), {res['n_cases']} cases, "
        f"{res['n_spans']} spans, {len(res['runs'])} runs",
        "",
        "| measure | mean | Wilson 95% | threshold | pass |",
        "|---|---|---|---|---|",
    ]
    for g in res["gate"]:
        ci = res["wilson95"].get(g["key"])
        cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        lines.append(
            f"| {g['measure']} | {f(g['value'])} | {cis} | {g['threshold']} | "
            f"{ {True: 'yes', False: 'NO', None: 'n/a'}[g['passed']] } |"
        )
    m = res["mean"]
    lines += [
        "",
        f"severity within-1: {f(m.get('severity_within1'))}; flip rates: target "
        f"{f(m.get('target_flip_rate'))}, severity {f(m.get('severity_flip_rate'))}, "
        f"unsupported_kind {f(m.get('unsupported_kind_flip_rate'))}",
        "",
    ]
    for fld, mat in res["confusion"].items():
        lines.append(
            f"confusion `{fld}` (gold -> pred counts, pooled over runs): {json.dumps(mat)}"
        )
    if "disagreements" in res:
        lines += ["", f"### Disagreements ({len(res['disagreements'])})", ""]
        for d in res["disagreements"]:
            diff = ", ".join(
                f"{k}: gold {v['gold']} / pred {v['pred']}" for k, v in d["diff"].items()
            )
            lines.append(
                f"- r{d['run']} {d['case_id']} {d['span_id']} [{d['grounding']}] "
                f"{json.dumps(d['text'], ensure_ascii=False)} — {diff} — why: {d['why']}"
            )
    return "\n".join(lines)


def run(
    split: str,
    *,
    model: str,
    effort: str | None,
    out_root: Path,
    repeats: int = 2,
    rpm: float = 1500.0,
    concurrency: int = 64,
    cases_path: Path | None = None,
) -> dict[str, Any]:
    from hallucination_bench.evals.wsbench_chat.run import model_slug

    cases = load_cases(split, cases_path)
    d = Path(out_root) / "calibration" / model_slug(model)
    usage = Usage()

    async def go() -> list[list[list[tv1.SpanLabel]]]:
        llm = LLM(rpm=rpm, concurrency=concurrency, effort=effort)
        try:
            with Cache(d / "cache.jsonl") as cache:
                return await classify_cases(
                    cases,
                    split=split,
                    model=model,
                    effort=effort,
                    cache=cache,
                    llm=llm,
                    repeats=repeats,
                    usage=usage,
                )
        finally:
            await llm.aclose()

    labels = asyncio.run(go())
    res = score(cases, labels, split)
    res["spend_this_run"] = usage.to_dict()
    res["model"] = model
    res["effort"] = effort
    (d / f"{split}_{tv1.PROMPT_VERSION}.json").write_text(json.dumps(res, indent=1) + "\n")
    (d / f"{split}_{tv1.PROMPT_VERSION}.md").write_text(render_markdown(res) + "\n")
    return res
