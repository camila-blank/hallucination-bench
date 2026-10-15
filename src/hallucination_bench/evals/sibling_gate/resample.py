"""Plan 0009 §2a: rung-1 resampling. Pools, trimmed manifests, generator job specs, the
block-shape check and the collect step.

Nothing here launches a job or writes a Modal volume: :func:`write_manifests` writes the trimmed
manifests and the volume copy plan under ``outputs/sibling_gate/resample/<pool>/``, and
:func:`job_lines` returns the shell lines ``scripts/sibling_gate/run_all.sh`` runs (from the
gw-mlnla script snapshot; the scripts are never edited).

* **Pools.** Arms that share a lens and checkpoint share one pool (one resample per cell). A
  pool's layer list is the union of its units' layers.
* **Trimmed manifest** (``acts-sibgate-<pool>``): the original manifest's ``model_id`` and
  ``layers`` (so ``ActsManifest.load`` and ``check_bank_layers`` pass), only the needed prompts,
  each entry's ``file`` re-pointed to a **copy** of its acts file inside the new subdir, and each
  entry's own ``eval_positions`` = the union of positions its units need (``--eval-positions``
  is never used: it overwrites every entry).
* **Over-generation.** The generators read every listed position at every listed layer of a
  prompt (units = prompts x layers), so ``n_generated = sum_prompts |positions| x |layers|``;
  collect keeps only the requested (id, layer, pos) keys.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from hallucination_bench import arms as _arms

from . import units as U

N_RESAMPLE = 10
ACTS_11 = "acts-hallucination-chat"
ACTS_L42 = "acts-hallucination-chat-L42"
ACTS_VOLUME = "wsbench-acts"
# PromptEntry fields (gw-mlnla scripts/oracle_lens_evals/olens_sglang/common.py): an entry with
# any other key fails ``ActsManifest.load``
PROMPT_ENTRY_FIELDS = frozenset(
    {
        "label",
        "family",
        "file",
        "n_pos",
        "tokens",
        "targets",
        "look_for",
        "eval_positions",
        "units",
        "contract",
    }
)
PROMPT_ENTRY_REQUIRED = frozenset({"label", "family", "file", "n_pos", "tokens", "targets"})
# §2a sampling contract (the original run_config.json): T 1.0, top_p 0.95, top_k 64, max_new 256
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new": 256}


@dataclass(frozen=True)
class Pool:
    name: str
    arms: tuple[str, ...]
    generator: str  # read | nla | patchscopes
    acts: str  # the source acts subdir
    lens: str = ""  # wsbench_nla_modal --lens
    conditional: bool = False  # patchscopes: resample_failed unless the pre-check passes
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def subdir(self) -> str:
        return f"acts-sibgate-{self.name}"

    @property
    def run(self) -> str:
        return f"sibgate-{self.name}"


_GEN_VIEW = {"lensagree": "patchscopes"}  # registry generator -> the plan-0009 Pool.generator


def pool_from_registry(name: str, reg: dict[str, Any] | None = None) -> Pool:
    """Any registered pool as a :class:`Pool` (``arms`` = its gated arms, else every registered
    arm of the pool)."""
    r = reg if reg is not None else _arms.load()
    row = r["pools"][name]
    gated = _arms.pools_of_gated(r).get(name) or _arms.pool_arms(name, r)
    args = row.get("args") or {}
    return Pool(
        name,
        tuple(gated),
        _GEN_VIEW.get(row["generator"], row["generator"]),
        args.get("acts", ACTS_11),
        lens=args.get("lens", ""),
        conditional=bool(row.get("conditional")),
    )


# Plan 0010 §1: views of the registry (pools serving a gated arm, registry order); a test asserts
# they equal the plan-0009 values.
POOLS: dict[str, Pool] = {p: pool_from_registry(p) for p in _arms.pools_of_gated()}
POOL_OF: dict[str, str] = {a: p.name for p in POOLS.values() for a in p.arms}


def pool_of(arm: str) -> str | None:
    """The arm's pool (``None`` for the deterministic token arms). Plan 0010: every registered
    arm resolves (a registered arm outside the gated set, e.g. a ``--add-arm`` candidate)."""
    return POOL_OF.get(arm) or _arms.pool_of(arm)


# ---------------------------------------------------------------- requests

Key = tuple[str, int, int]  # (id, layer, pos)


@dataclass
class PoolRequest:
    pool: str
    cells: set[Key] = field(default_factory=set)
    n_units: int = 0

    @property
    def layers(self) -> list[int]:
        return sorted({la for _i, la, _p in self.cells})

    def positions(self) -> dict[str, list[int]]:
        """Per prompt: the union of positions its units need (every layer)."""
        out: dict[str, set[int]] = defaultdict(set)
        for i, _la, p in self.cells:
            out[i].add(p)
        return {i: sorted(ps) for i, ps in sorted(out.items())}

    @property
    def n_requested(self) -> int:
        return len(self.cells)

    @property
    def n_generated(self) -> int:
        """Cells the generator produces: every listed position at every listed layer."""
        nl = len(self.layers)
        return sum(len(ps) * nl for ps in self.positions().values())

    def counts(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "n_units": self.n_units,
            "n_prompts": len(self.positions()),
            "layers": self.layers,
            "n_requested_cells": self.n_requested,
            "n_generated_cells": self.n_generated,
            "n_generated_samples": self.n_generated * N_RESAMPLE,
        }


def requests(units: Iterable[Mapping[str, Any]]) -> dict[str, PoolRequest]:
    """Pool -> the (id, layer, pos) cells holding at least one unit (deterministic arms skipped)."""
    out: dict[str, PoolRequest] = {}
    for u in units:
        p = pool_of(u["arm"])
        if p is None:
            continue
        r = out.setdefault(p, PoolRequest(p))
        r.cells.add((u["id"], int(u["layer"]), int(u["pos"])))
        r.n_units += 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- trimmed manifests


# The L42 manifest's files point at ``acts-jlens-pr-L42``, which no longer exists on the volume;
# the same 149 prompts' layer-42 acts (identical tokens and eval_positions) live in -L42cap.
DIR_REMAP = {"acts-jlens-pr-L42": "acts-hallucination-chat-L42cap"}


def _vol_path(file: str, src_subdir: str) -> PurePosixPath:
    """A manifest ``file`` -> its path inside the acts volume (``/acts/x`` mounts the root)."""
    p = PurePosixPath(file)
    if p.is_absolute():
        parts = p.parts[2:] if len(p.parts) > 2 and p.parts[1] == "acts" else p.parts[1:]
        if parts and parts[0] in DIR_REMAP:
            parts = (DIR_REMAP[parts[0]], *parts[1:])
        return PurePosixPath("/", *parts)
    return PurePosixPath("/", src_subdir, *p.parts)


def trimmed_manifest(
    orig: Mapping[str, Any], pool: Pool, positions: Mapping[str, list[int]]
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """(manifest for ``acts-sibgate-<pool>``, [(source volume path, copy volume path)])."""
    ents = {e["label"]: e for e in orig["prompts"]}
    missing = sorted(set(positions) - set(ents))
    if missing:
        raise ValueError(
            f"{pool.name}: {len(missing)} prompts not in the source manifest, e.g. {missing[:3]}"
        )
    prompts, copies = [], []
    for label, pos in positions.items():
        e = dict(ents[label])
        bad = [p for p in pos if not 0 <= p < int(e["n_pos"])]
        if bad:
            raise ValueError(f"{label}: positions {bad} outside n_pos={e['n_pos']}")
        src = _vol_path(e["file"], pool.acts)
        dst = PurePosixPath("/", pool.subdir, src.name)
        copies.append((str(src), str(dst)))
        e["file"] = src.name  # relative: resolves inside the (copied) subdir
        e["eval_positions"] = sorted(int(p) for p in pos)
        prompts.append(e)
    man = {"model_id": orig["model_id"], "layers": list(orig["layers"]), "prompts": prompts}
    return man, copies


def check_manifest(man: Mapping[str, Any], layers: Iterable[int]) -> list[str]:
    """The checks ``ActsManifest.load`` + ``check_bank_layers`` + ``effective_positions`` apply,
    re-stated here (the gw-mlnla code needs torch): empty list = OK."""
    errs = []
    for k in ("model_id", "layers", "prompts"):
        if k not in man:
            errs.append(f"manifest lacks {k}")
    if errs:
        return errs
    have = {int(x) for x in man["layers"]}
    miss = sorted(set(int(x) for x in layers) - have)
    if miss:
        errs.append(f"no plane at layer(s) {miss} (bank layers {sorted(have)})")
    for e in man["prompts"]:
        extra = set(e) - PROMPT_ENTRY_FIELDS
        lack = PROMPT_ENTRY_REQUIRED - set(e)
        if extra or lack:
            errs.append(
                f"{e.get('label')}: PromptEntry fields extra={sorted(extra)} missing={sorted(lack)}"
            )
        ep = e.get("eval_positions")
        if ep is not None and any(not 0 <= int(p) < int(e["n_pos"]) for p in ep):
            errs.append(f"{e.get('label')}: eval_positions outside n_pos")
    return errs


def expected_keys(man: Mapping[str, Any], layers: Iterable[int]) -> set[Key]:
    """Every (id, layer, pos) the generator writes for ``man`` at ``layers``."""
    ls = sorted(int(x) for x in layers)
    out = set()
    for e in man["prompts"]:
        ps = e.get("eval_positions")
        ps = range(int(e["n_pos"])) if ps is None else ps
        out |= {(e["label"], la, int(p)) for la in ls for p in ps}
    return out


def pool_dir(pool: str, out: Path = U.OUT) -> Path:
    return Path(out) / "resample" / pool


def write_manifests(
    reqs: Mapping[str, PoolRequest],
    orig_manifests: Mapping[str, Mapping[str, Any]],
    out: Path = U.OUT,
    *,
    limit_cells: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Write ``manifest.json``, ``copies.json``, ``request.json`` and ``jobs.sh`` per pool
    (local files only). ``orig_manifests``: source acts subdir -> its manifest.
    ``limit_cells`` (step 0 pre-check: 5) trims each pool to its first requested cells."""
    res = {}
    for name, r in reqs.items():
        pool = POOLS[name]
        if limit_cells is not None:
            r = PoolRequest(name, set(sorted(r.cells)[:limit_cells]), r.n_units)
        man, copies = trimmed_manifest(orig_manifests[pool.acts], pool, r.positions())
        errs = check_manifest(man, r.layers)
        d = pool_dir(name, out) / ("precheck" if limit_cells is not None else "")
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(man))
        (d / "copies.json").write_text(json.dumps(copies, indent=0))
        (d / "request.json").write_text(
            json.dumps({**r.counts(), "cells": sorted(r.cells)}, indent=0)
        )
        (d / "jobs.sh").write_text("\n".join(job_lines(pool, r.layers, d, copies)) + "\n")
        res[name] = {**r.counts(), "manifest_errors": errs, "dir": str(d)}
    return res


# ---------------------------------------------------------------- job specs (shell lines)

READ_SCRIPT = "scripts/olens_suite/workspace_bench/wsbench_read_modal.py::main"  # two+ entrypoints
NLA_SCRIPT = "scripts/olens_suite/workspace_bench/wsbench_nla_modal.py"
PATCHSCOPES_SCRIPT = "scripts/patchscopes/modal_lensagree.py"
MODAL = "uvx --python 3.12 modal@latest"


def job_lines(pool: Pool, layers: list[int], local_dir: Path, copies: list[tuple[str, str]]):
    """The shell lines that seed ``acts-sibgate-<pool>`` and run the pool's generator from the
    script snapshot (``$SNAP``, a copy of /workspace/gw-mlnla at a recorded HEAD). ``$MERGED_*``
    names the original run's merged dir (from its ``run_config.json``)."""
    lay = ",".join(str(x) for x in layers)
    lines = [
        f"# pool {pool.name}: arms {', '.join(pool.arms)}; layers {lay}; generator "
        f"{pool.generator}",
        f"{MODAL} volume put --force {ACTS_VOLUME} {local_dir / 'manifest.json'} "
        f"/{pool.subdir}/manifest.json",
    ]
    lines += [f"{MODAL} volume cp {ACTS_VOLUME} {a} {b}" for a, b in copies]
    s = SAMPLING
    if pool.generator == "read":
        var = "MERGED_" + pool.name.upper().replace("-", "_")
        lines.append(
            f"(cd $SNAP && PYTHONPATH=src {MODAL} run {READ_SCRIPT} --skip-merge "
            f"--merged-name ${var} --run {pool.run} --acts-subdir {pool.subdir} --layers {lay} "
            f"--k {N_RESAMPLE} --temperature {s['temperature']} --top-p {s['top_p']} "
            f"--top-k {s['top_k']} --max-new {s['max_new']} --shards 16)"
        )
    elif pool.generator == "nla":
        lines.append(
            f"(cd $SNAP && PYTHONPATH=src {MODAL} run {NLA_SCRIPT} --lens {pool.lens} "
            f"--acts-subdir {pool.subdir} --layers {lay} --run {pool.run} "
            f"--n-samples {N_RESAMPLE} --temperature {s['temperature']} --top-p {s['top_p']} "
            f"--top-k {s['top_k']} --shards 16)"
        )
    else:  # patchscopes: k=1 per call in its own config; conditional on the step-0 pre-check
        lines.append(
            "# patchscopes is conditional (§2a): run only if the step-0 pre-check passes; its "
            "entrypoint draws one sample per cell, so the 10 draws are 10 seeded runs"
        )
        for sd in range(N_RESAMPLE):
            lines.append(
                f"(cd $SNAP && PYTHONPATH=src {MODAL} run {PATCHSCOPES_SCRIPT} "
                f"--acts-subdir {pool.subdir} --run {pool.run}-s{sd} --source-layers {lay} "
                f"--seed {sd} --shards 16)"
            )
    return lines


# ---------------------------------------------------------------- collect


def iter_gen_rows(src: Path) -> Iterator[dict[str, Any]]:
    """Readout-contract rows from a converted ``hallucination.jsonl`` (``{id, layer, pos,
    token, samples}``) or a raw gen dir (``<label>/L{layer:03d}.jsonl`` rows with ``label``)."""
    src = Path(src)
    files = (
        [src]
        if src.is_file()
        else sorted(  # ``volume get`` nests each run dir: <src>/<run>/<label>/L*.jsonl
            f
            for f in src.rglob("L*.jsonl")
            if not any(x.startswith("_") for x in f.relative_to(src).parts[:-1])
        )
    )  # "_work/" holds the patchscopes shard stores (a different schema)
    for f in files:
        for r in U.iter_jsonl(f):
            yield {
                "id": r.get("id") or r.get("label"),
                "layer": int(r["layer"]),
                "pos": int(r["pos"]),
                "token": r.get("token"),
                "samples": list(r.get("samples") or []),
            }


def collect(
    rows: Iterable[Mapping[str, Any]],
    requested: Iterable[Key],
    config: Mapping[str, Any],
    *,
    n: int = N_RESAMPLE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only the requested keys (over-generated cells are discarded); several runs of one
    key (patchscopes seeds) are concatenated in order. A key with fewer than ``n`` samples is
    reported as short (its units become ``resample_failed``)."""
    want = set(requested)
    got: dict[Key, dict[str, Any]] = {}
    n_seen = n_dropped = 0
    for r in rows:
        n_seen += 1
        k = (r["id"], int(r["layer"]), int(r["pos"]))
        if k not in want:
            n_dropped += 1
            continue
        g = got.setdefault(
            k, {"id": k[0], "layer": k[1], "pos": k[2], "token": r.get("token"), "samples": []}
        )
        g["samples"].extend(r["samples"])
    out, short = [], []
    for k in sorted(got):
        g = got[k]
        if len(g["samples"]) < n:
            short.append(k)
            continue
        out.append({**g, "samples": g["samples"][:n], "config": dict(config)})
    missing = sorted(want - set(got))
    stats = {
        "n_requested": len(want),
        "n_rows_seen": n_seen,
        "n_dropped_overgenerated": n_dropped,
        "n_collected": len(out),
        "n_short": len(short),
        "n_missing": len(missing),
        "missing_sample": [list(x) for x in missing[:5]],
        "keys_equal_request": not missing and not short,
    }
    return out, stats


def resample_path(pool: str, out: Path = U.OUT) -> Path:
    return pool_dir(pool, out) / "resample.jsonl"


def load_resample(pool: str, out: Path = U.OUT) -> dict[Key, list[str]]:
    p = resample_path(pool, out)
    if not p.exists():
        return {}
    return {(r["id"], int(r["layer"]), int(r["pos"])): r["samples"] for r in U.iter_jsonl(p)}


# ---------------------------------------------------------------- block shape (step 0)


def n_list_lines(block: str) -> int:
    return sum(1 for line in block.splitlines() if line.lstrip().startswith("- "))


def block_shape(samples: Iterable[str]) -> dict[str, Any]:
    ns, ls = [], []
    for s in samples:
        ns.append(n_list_lines(s))
        ls.append(len(s))
    if not ns:
        return {"n": 0}
    q = statistics.quantiles(ls, n=4) if len(ls) > 1 else [ls[0]] * 3
    return {
        "n": len(ns),
        "lines_mode": Counter(ns).most_common(1)[0][0],
        "lines_mean": statistics.fmean(ns),
        "len_q1": q[0],
        "len_median": statistics.median(ls),
        "len_q3": q[2],
    }


# pre-registered tolerances of the block-shape check (step 0; the plan states the quantities,
# these thresholds are the implementation's)
SHAPE_LINES_MEAN_TOL = 0.5
SHAPE_MEDIAN_RATIO = (0.8, 1.25)


def block_shape_check(orig: Iterable[str], new: Iterable[str]) -> dict[str, Any]:
    """The number of "- " lines per sample and the length distribution must match the originals:
    same modal line count, mean line count within 0.5, median length within x0.8..x1.25."""
    a, b = block_shape(orig), block_shape(new)
    if not a["n"] or not b["n"]:
        return {"ok": False, "orig": a, "new": b, "why": "empty"}
    ratio = b["len_median"] / a["len_median"] if a["len_median"] else float("inf")
    ok = (
        a["lines_mode"] == b["lines_mode"]
        and abs(a["lines_mean"] - b["lines_mean"]) <= SHAPE_LINES_MEAN_TOL
        and SHAPE_MEDIAN_RATIO[0] <= ratio <= SHAPE_MEDIAN_RATIO[1]
    )
    return {"ok": ok, "orig": a, "new": b, "median_ratio": ratio}


# ================================================================ plan 0010 §2: recoverable runs
#
# The volume is the source of truth. Each launch is a run ``gen-sibgate-<pool>-<sha8>-a<att>``
# (``-r<j>`` for pools with ``samples_per_run`` < 10, one seeded run per missing sample); sha8 =
# the hash of the sorted requested keys. A key's sample count is the sum over runs (run0 = the
# imported plan-0009 ``resample.jsonl``, first; then attempt order); complete at >= 10, cut to
# the first 10. Plan-0009 names never match :data:`RUN_RE`, so legacy runs are read via run0 only.

RUN_RE = re.compile(
    r"^gen-sibgate-(?P<pool>.+?)-(?P<sha>[0-9a-f]{8})-a(?P<att>\d+)(?:-r(?P<r>\d+))?$"
)
RUN0 = "run0"
MAX_ATTEMPTS = 2


def keys_sha8(keys: Iterable[Key]) -> str:
    blob = json.dumps(sorted([k[0], int(k[1]), int(k[2])] for k in keys)).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def run_name(pool: str, keys: Iterable[Key], attempt: int, r: int | None = None) -> str:
    base = f"gen-sibgate-{pool}-{keys_sha8(keys)}-a{int(attempt)}"
    return base if r is None else f"{base}-r{int(r)}"


def parse_run_name(name: str, pools: Iterable[str]) -> dict[str, Any] | None:
    """The anchored parse; the pool must equal a registry pool exactly (``None`` otherwise)."""
    m = RUN_RE.match(name)
    if m is None or m.group("pool") not in set(pools):
        return None
    return {
        "pool": m.group("pool"),
        "sha": m.group("sha"),
        "attempt": int(m.group("att")),
        "r": None if m.group("r") is None else int(m.group("r")),
    }


def seed_of(name: str) -> int:
    """Seed of a per-sample run: ``int(sha256(run_name)[:8], 16)`` (distinct across runs)."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def run_order(name: str) -> tuple[int, int, str, int]:
    """run0 first, then attempt order (then sha, then the per-sample index)."""
    if name == RUN0:
        return (0, 0, "", 0)
    m = RUN_RE.match(name)
    if m is None:
        return (2, 0, name, 0)
    return (1, int(m.group("att")), m.group("sha"), int(m.group("r") or 0))


@dataclass
class PoolState:
    """``outputs/sibling_gate/resample/<pool>/state.json``: the launched runs and, per key, the
    attempts that requested it. Collected rows live in ``runs/<run>.jsonl`` (one file per run, so
    a re-collected run is never double-counted)."""

    pool: str
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)  # "id|layer|pos" -> attempts

    @staticmethod
    def kstr(k: Key) -> str:
        return f"{k[0]}|{int(k[1])}|{int(k[2])}"

    def n_attempts(self, k: Key) -> int:
        return int(self.attempts.get(self.kstr(k), 0))

    def to_dict(self) -> dict[str, Any]:
        return {"pool": self.pool, "runs": self.runs, "attempts": self.attempts}


def state_path(pool: str, out: Path = U.OUT) -> Path:
    return pool_dir(pool, out) / "state.json"


def runs_dir(pool: str, out: Path = U.OUT) -> Path:
    return pool_dir(pool, out) / "runs"


def load_state(pool: str, out: Path = U.OUT) -> PoolState:
    p = state_path(pool, out)
    if not p.exists():
        return PoolState(pool)
    d = json.loads(p.read_text())
    return PoolState(pool, d.get("runs") or {}, d.get("attempts") or {})


def save_state(st: PoolState, out: Path = U.OUT) -> None:
    p = state_path(st.pool, out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st.to_dict(), indent=0) + "\n")
    tmp.replace(p)


def collected_runs(pool: str, out: Path = U.OUT) -> list[str]:
    d = runs_dir(pool, out)
    return sorted((f.stem for f in d.glob("*.jsonl")), key=run_order) if d.exists() else []


def merge_runs(
    runs: Mapping[str, Iterable[Mapping[str, Any]]], n: int = N_RESAMPLE
) -> tuple[dict[Key, dict[str, Any]], dict[Key, int]]:
    """(complete keys -> row with the first ``n`` samples, every key -> its summed count).
    Runs in :func:`run_order`; within a run a key's rows are deduplicated by (run, seed): a run
    contributes one sample block per (key, seed)."""
    got: dict[Key, dict[str, Any]] = {}
    for name in sorted(runs, key=run_order):
        seen: set[tuple[Key, Any]] = set()
        for r in runs[name]:
            k = (r["id"], int(r["layer"]), int(r["pos"]))
            sd = r.get("seed")
            if sd is not None:  # per-sample runs: one block per (run, seed)
                if (k, sd) in seen:
                    continue
                seen.add((k, sd))
            g = got.setdefault(
                k,
                {
                    "id": k[0],
                    "layer": k[1],
                    "pos": k[2],
                    "token": r.get("token"),
                    "samples": [],
                    "config": r.get("config"),
                    "runs": [],
                },
            )
            g["samples"].extend(r.get("samples") or [])
            if name not in g["runs"]:
                g["runs"].append(name)
    counts = {k: len(g["samples"]) for k, g in got.items()}
    complete = {
        k: {**g, "samples": g["samples"][:n]} for k, g in got.items() if len(g["samples"]) >= n
    }
    return complete, counts


def load_runs(pool: str, out: Path = U.OUT) -> dict[str, list[dict[str, Any]]]:
    return {
        r: list(U.iter_jsonl(runs_dir(pool, out) / f"{r}.jsonl")) for r in collected_runs(pool, out)
    }


def rebuild_resample(pool: str, out: Path = U.OUT, config: Mapping[str, Any] | None = None):
    """Rewrite ``resample.jsonl`` (what fill reads) from every collected run; returns counts."""
    complete, counts = merge_runs(load_runs(pool, out))
    cfg = dict(config or {"pool": pool})
    rows = []
    for k in sorted(complete):
        g = complete[k]
        rows.append(
            {
                "id": g["id"],
                "layer": g["layer"],
                "pos": g["pos"],
                "token": g["token"],
                "samples": g["samples"],
                "config": g.get("config") or cfg,
            }
        )
    U.write_jsonl(resample_path(pool, out), rows)
    return complete, counts


@dataclass
class Launch:
    """One planned generator run."""

    pool: str
    run: str  # the volume dir name gen-sibgate-...
    attempt: int
    keys: list[Key]
    seed: int | None = None

    @property
    def run_arg(self) -> str:  # the generator's --run (it prefixes "gen-")
        return self.run[len("gen-") :]


def plan_launches(
    pool: str,
    want: Iterable[Key],
    counts: Mapping[Key, int],
    st: PoolState,
    samples_per_run: int,
    n: int = N_RESAMPLE,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[list[Launch], list[Key]]:
    """(launches of the next attempt, keys that are exhausted = ``resample_failed``). Requested
    = wanted keys minus complete keys minus keys with ``max_attempts`` attempts."""
    need = sorted(k for k in set(want) if counts.get(k, 0) < n)
    failed = [k for k in need if st.n_attempts(k) >= max_attempts]
    req = [k for k in need if st.n_attempts(k) < max_attempts]
    if not req:
        return [], failed
    att = 1 + max(st.n_attempts(k) for k in req)
    if samples_per_run >= n:
        return [Launch(pool, run_name(pool, req, att), att, req)], failed
    k_runs = max(-(-(n - counts.get(k, 0)) // samples_per_run) for k in req)
    out = []
    for j in range(k_runs):
        name = run_name(pool, req, att, j)
        out.append(Launch(pool, name, att, req, seed=seed_of(name)))
    return out, failed


def generator_cmd(
    pool_row: Mapping[str, Any], launch: Launch, subdir: str, layers: list[int], modal: str
) -> str:
    """The detached generator command for one run (run from the gw-mlnla snapshot)."""
    lay = ",".join(str(x) for x in layers)
    s = pool_row.get("sampling") or SAMPLING
    args = pool_row.get("args") or {}
    g = pool_row["generator"]
    k = int(pool_row.get("samples_per_run") or N_RESAMPLE)
    if g == "read":
        return (
            f"PYTHONPATH=src {modal} run --detach {READ_SCRIPT} --skip-merge "
            f"--merged-name {args['merged']} --run {launch.run_arg} --acts-subdir {subdir} "
            f"--layers {lay} --k {k} --temperature {s['temperature']} --top-p {s['top_p']} "
            f"--top-k {s['top_k']} --max-new {s['max_new']} --shards 16"
        )
    if g == "nla":
        return (
            f"PYTHONPATH=src {modal} run --detach {NLA_SCRIPT} --lens {args['lens']} "
            f"--acts-subdir {subdir} --layers {lay} --run {launch.run_arg} --n-samples {k} "
            f"--temperature {s['temperature']} --top-p {s['top_p']} --top-k {s['top_k']} "
            "--shards 16"
        )
    if g == "lensagree":
        return (
            f"PYTHONPATH=src {modal} run --detach {PATCHSCOPES_SCRIPT} --acts-subdir {subdir} "
            f"--run {launch.run_arg} --source-layers {lay} --seed {launch.seed} --shards 16"
        )
    raise ValueError(f"unknown generator {g}")


def collect_run_rows(
    src: Path, launch_keys: Iterable[Key], seed: int | None = None
) -> list[dict[str, Any]]:
    """A fetched run dir -> its rows for the launch's keys (over-generated cells dropped)."""
    want = set(launch_keys)
    out = []
    for r in iter_gen_rows(src):
        k = (r["id"], int(r["layer"]), int(r["pos"]))
        if k in want:
            out.append({**r, **({"seed": seed} if seed is not None else {})})
    return out


# ---------------------------------------------------------------- Modal operations (injectable)

FATAL_PATTERNS = (
    "spend limit",
    "spending limit",
    "billing",
    "quota",
    "unauthorized",
    "authentication",
    "token is invalid",
    "not authenticated",
    "permission denied",
)


class ModalFatal(RuntimeError):
    """A spend-limit / auth / quota error (``hbench gate`` exits 3)."""


def fatal_reason(text: str) -> str | None:
    low = text.lower()
    for p in FATAL_PATTERNS:
        if p in low:
            return p
    return None


class ModalOps:
    """The only path to Modal (tests pass a fake). ``n_launches`` counts generator / capture /
    fit / score jobs; volume ``ls`` / ``get`` / ``put`` / ``cp`` are not launches."""

    def __init__(self, modal: str, env: Mapping[str, str], cwd: Path | None = None) -> None:
        self.modal = modal
        self.env = dict(env)
        self.cwd = cwd
        self.n_launches = 0
        self.n_volume_ops = 0
        self.log: list[str] = []

    def _env(self) -> dict[str, str]:
        import os

        return {**os.environ, **self.env}

    def sh(self, cmd: str, cwd: Path | None = None) -> tuple[int, str]:
        import subprocess

        self.log.append(cmd)
        p = subprocess.run(
            cmd, shell=True, cwd=cwd or self.cwd, env=self._env(), capture_output=True, text=True
        )
        txt = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0 and (why := fatal_reason(txt)):
            raise ModalFatal(f"{why}: {txt.strip()[-800:]}")
        return p.returncode, txt

    def volume(self, args: str) -> tuple[int, str]:
        self.n_volume_ops += 1
        return self.sh(f"{self.modal} volume {args}")

    def ls(self, volume: str, path: str = "/") -> list[str]:
        rc, txt = self.volume(f"ls {volume} {path}")
        if rc != 0:
            return []
        names = []
        for line in txt.splitlines():
            t = line.strip().strip("│┃|").strip()
            if t:
                names.append(t.split()[0].rstrip("/").split("/")[-1])
        return names

    def get(self, volume: str, remote: str, local: Path) -> bool:
        Path(local).mkdir(parents=True, exist_ok=True)
        rc, _ = self.volume(f"get --force {volume} {remote} {local}/")
        return rc == 0

    def start(self, cmd: str, cwd: Path, log: Path):
        """Start a detached job (``modal run --detach``); returns the Popen handle."""
        import subprocess

        self.n_launches += 1
        self.log.append(cmd)
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("a")
        return subprocess.Popen(
            cmd, shell=True, cwd=cwd, env=self._env(), stdout=fh, stderr=subprocess.STDOUT
        )

    def run(self, cmd: str, cwd: Path | None = None) -> tuple[int, str]:
        """A blocking job (fit / score / capture)."""
        self.n_launches += 1
        return self.sh(cmd, cwd)
