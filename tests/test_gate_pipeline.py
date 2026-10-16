"""Plan 0010: the sibling gate as a pipeline step (``hbench gate``). No LLM, no Modal: a fake LLM
answers every gate prompt and a fake Modal plays the generator / capture / fit / score jobs and
the gen volume, so the whole gate runs end to end on a toy world (two real bank items)."""

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pytest

from hallucination_bench import arms as A
from hallucination_bench import report_v2
from hallucination_bench.cache import Cache
from hallucination_bench.evals.sibling_gate import classes as K
from hallucination_bench.evals.sibling_gate import fill_judge as F
from hallucination_bench.evals.sibling_gate import gate as G
from hallucination_bench.evals.sibling_gate import ladder as L
from hallucination_bench.evals.sibling_gate import resample as R
from hallucination_bench.evals.sibling_gate import texts as T
from hallucination_bench.evals.sibling_gate import units as U
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.llm import CallResult, Usage

SIB = "substitution_sibling"
ITEMS = ["chat-dailydialog-0000", "chat-dailydialog-0001"]
L11 = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]

# ---------------------------------------------------------------- 1. registry views


def test_registry_views_equal_plan_0009():
    prose = (
        "s3d-rl600",
        "s3d-rl600-all11",
        "s3d-sft251",
        "mlnla-rl600-all11",
        "mlnla-rl2-800-all11",
        "nla-rl-L42",
        "nla-sft-L42",
        "patchscopes",
    )
    tok = ("jlens", "logit", "tuned", "rlens", "template-v3", "sae-l42")
    # the plan-0009 arms come first; `gate --add-arm` appends later arms (e.g. mlnla-ck2000)
    p9 = set(prose) | set(tok)
    assert tuple(a for a in U.PROSE_ARMS if a in p9) == prose and U.TOKEN_ARMS == tok
    assert U.ARMS_IN_SCOPE[: len(prose) + len(tok)] == (*prose, *tok) or tuple(
        a for a in U.ARMS_IN_SCOPE if a in p9
    ) == (*prose, *tok)
    assert frozenset(tok) == U.DETERMINISTIC_ARMS
    fam = {a: a.split("-")[0] for a in prose if a.split("-")[0] in ("s3d", "mlnla", "nla")}
    for a in (*prose, *tok):
        assert U.family_of(a) == fam.get(a, a)
    want = {
        "s3d-rl600": (("s3d-rl600", "s3d-rl600-all11"), "read", R.ACTS_11, "", False),
        "s3d-sft251": (("s3d-sft251",), "read", R.ACTS_11, "", False),
        "mlnla-rl600": (("mlnla-rl600-all11",), "nla", R.ACTS_11, "mlnla-rl600", False),
        "mlnla-rl2-800": (("mlnla-rl2-800-all11",), "nla", R.ACTS_11, "mlnla-rl2-800", False),
        "nla-rl": (("nla-rl-L42",), "nla", R.ACTS_L42, "nla-rl400", False),
        "nla-sft": (("nla-sft-L42",), "nla", R.ACTS_L42, "nla-sft3864", False),
        "patchscopes": (("patchscopes",), "patchscopes", R.ACTS_11, "", True),
    }
    assert [p for p in R.POOLS if p in want] == list(want)  # later --add-arm pools may follow
    for p, (arms, gen, acts, lens, cond) in want.items():
        pl = R.POOLS[p]
        assert (pl.arms, pl.generator, pl.acts, pl.lens, pl.conditional) == (
            arms,
            gen,
            acts,
            lens,
            cond,
        )
    p9 = {a: p for p, v in want.items() for a in v[0]}
    assert {a: p for a, p in R.POOL_OF.items() if a in p9} == p9
    assert {p: a for p, a in A.arm_of_pool().items() if p in want} == {
        "s3d-rl600": "s3d-rl600-all11",
        "s3d-sft251": "s3d-sft251",
        "mlnla-rl600": "mlnla-rl600-all11",
        "mlnla-rl2-800": "mlnla-rl2-800-all11",
        "nla-rl": "nla-rl-L42",
        "nla-sft": "nla-sft-L42",
        "patchscopes": "patchscopes",
    }
    assert "declare -A ARM_OF=([s3d-rl600]=s3d-rl600-all11" in A.bash_views()
    # the validation arm is registered (plan 0010 §8 step 2 then appends it to gated_arms via
    # `gate --add-arm`); its pool resolves either way, and the plan-0009 arms stay first
    reg = A.load()
    assert "mlnla-ck2000-all11" in reg["arms"]
    assert reg["gated_arms"][:14] == [a for a in reg["gated_arms"] if a != "mlnla-ck2000-all11"]
    assert R.pool_of("mlnla-ck2000-all11") == "mlnla-ck2000"
    p = reg["pools"]["mlnla-ck2000"]
    assert (p["generator"], p["args"]["lens"], p["args"]["acts"]) == (
        "nla",
        "mlnla-rl2-2000",
        R.ACTS_11,
    )
    assert reg["pools"]["patchscopes"]["samples_per_run"] == 1
    assert reg["pools"]["s3d-rl600"]["orig_arm"] == "s3d-rl600-all11"


def test_unregistered_or_poolless_sampled_arm_exits_3(tmp_path):
    reg = A.load()
    with pytest.raises(A.RegistryError):
        A.with_added(reg, ["no-such-arm"])
    reg["arms"]["poolless"] = {"cells": None, "kind": "sampled", "family": "x", "pool": None}
    reg["gated_arms"].append("poolless")
    g = G.Gate(reg=reg, out=tmp_path / "o", data=tmp_path / "d", quiet=True)
    with pytest.raises(G.GateExit) as e:
        g.check_registry()
    assert e.value.code == G.EXIT_FATAL
    g = G.Gate(reg=reg, out=tmp_path / "o", data=tmp_path / "d", quiet=True, rung1_skip=True)
    g.check_registry()
    assert g.skipped_rung1 == ["poolless"] and "poolless" in g.deterministic


# ---------------------------------------------------------------- 2. runs


def test_run_names_attempts_and_regex():
    keys = [("i", 44, 3), ("i", 48, 3)]
    n1 = R.run_name("mlnla-rl2-800", keys, 1)
    assert re.fullmatch(r"gen-sibgate-mlnla-rl2-800-[0-9a-f]{8}-a1", n1)
    assert R.run_name("mlnla-rl2-800", keys, 2) != n1  # a retry gets a new name
    pools = ["mlnla-rl2-800", "mlnla-rl600", "patchscopes", "s3d-rl600"]
    assert R.parse_run_name(n1, pools)["pool"] == "mlnla-rl2-800"
    assert R.parse_run_name(n1, ["mlnla-rl2"]) is None  # the pool must match exactly
    # plan-0009 names never match (read via run0 only)
    for legacy in (
        "gen-sibgate-s3d-rl600",
        "gen-sibgate-patchscopes-s3",
        "gen-sibgate-s3d-rl600-precheck",
        "gen-sibgate-patchscopes-precheck-s0",
    ):
        assert R.parse_run_name(legacy, pools) is None


def _rows(keys, n, tag, seed=None):
    return [
        {"id": k[0], "layer": k[1], "pos": k[2], "samples": [f"{tag}{j}" for j in range(n)]}
        | ({"seed": seed} if seed is not None else {})
        for k in keys
    ]


def test_plan_0009_collision_partial_then_full_request_has_no_missing_keys():
    a, b = ("i", 44, 3), ("i", 44, 9)
    # the pilot bug: a partial run left b with 4 samples
    runs = {R.RUN0: _rows([a], 10, "r0-") + _rows([b], 4, "r0-")}
    complete, counts = R.merge_runs(runs)
    assert a in complete and b not in complete and counts[b] == 4
    st = R.PoolState("s3d-rl600")
    launches, failed = R.plan_launches("s3d-rl600", [a, b], counts, st, 10)
    assert [x.keys for x in launches] == [[b]] and not failed
    assert launches[0].run.endswith("-a1")
    runs[launches[0].run] = _rows([b], 10, "a1-")
    complete, counts = R.merge_runs(runs)
    assert set(complete) == {a, b}  # no missing keys
    assert complete[b]["samples"][:4] == ["r0-0", "r0-1", "r0-2", "r0-3"]  # run0 first
    assert len(complete[b]["samples"]) == 10 and counts[b] == 14
    # still short after 2 attempts -> resample_failed (no relaunch)
    st.attempts[st.kstr(b)] = 2
    launches, failed = R.plan_launches("s3d-rl600", [a, b], {a: 10, b: 4}, st, 10)
    assert launches == [] and failed == [b]


def test_patchscopes_seed_runs_add_up_to_ten_and_never_double_count():
    k = ("i", 44, 3)
    st = R.PoolState("patchscopes")
    launches, _ = R.plan_launches("patchscopes", [k], {}, st, 1)
    assert len(launches) == 10
    assert [x.run[-3:] for x in launches] == [f"-r{j}" for j in range(10)]
    seeds = [x.seed for x in launches]
    assert len(set(seeds)) == 10 and seeds[0] == R.seed_of(launches[0].run)
    runs = {x.run: _rows([k], 1, f"{x.run}-", seed=x.seed) for x in launches[:9]}
    runs[launches[0].run] += _rows([k], 1, "dup-", seed=launches[0].seed)  # re-collected
    complete, counts = R.merge_runs(runs)
    assert counts[k] == 9 and k not in complete
    # attempt 2 asks only the missing sample, with seeds distinct from attempt 1
    st.attempts[st.kstr(k)] = 1
    l2, _ = R.plan_launches("patchscopes", [k], counts, st, 1)
    assert len(l2) == 1 and l2[0].attempt == 2 and l2[0].seed not in seeds
    runs[l2[0].run] = _rows([k], 1, "a2-", seed=l2[0].seed)
    complete, counts = R.merge_runs(runs)
    assert counts[k] == 10 and len(complete[k]["samples"]) == 10


# ---------------------------------------------------------------- the toy world (fakes)

SAIDS = ["Google", "Bing", "Yahoo", "Baidu", "Yandex"]


def seg(readout, start, n, said, actual="the engineers"):
    return {
        "unit": f"R{readout}.U{start}",
        "readout": readout,
        "start": start,
        "end": start + n,
        "text": f"{said} built me",
        "leaf": SIB,
        "raw_leaf": SIB,
        "status": "ok",
        "alt_leaf": None,
        "pair": {"said": said, "actual": actual},
        "contradicted_by": None,
    }


def cell(item, layer, pos, saids):
    key = U.cell_key(item, layer, pos)
    return {
        "key": key,
        "id": item,
        "layer": layer,
        "pos": pos,
        "stages": {"map": "ok"},
        "map": {"segments": [seg(0, 20 * i, 10, s) for i, s in enumerate(saids)]},
    }


def sites():
    its = {it["id"]: it for it in bank.load_items()}
    return {i: [int(s["pos"]) for s in its[i]["sites"]] for i in ITEMS}


def write_cells(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


BLOCK = "- {s} is the builder {j}\n- second line\n- third line\n- fourth line"


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.n = 0

    async def call(self, *, model, system, user, schema):
        self.calls.append(system[:40])
        self.n += 1
        if system == F.SYSTEM:
            nb = int(re.search(r"(\d+) FRESH READOUT BLOCKS", user).group(1))
            blocks = re.findall(r"=== block \d+ ===\n(.*?)(?=\n\n=== block|\Z)", user, re.S)
            out = [
                {"block": i + 1, "answer": "said", "objects": [], "quote": b[2:12]}
                for i, b in enumerate(blocks[:nb])
            ]
            return CallResult({"blocks": out, "note": ""}, Usage(usd=0.01, calls=1))
        if system == K.CLASS_SYSTEM:
            said = json.loads(re.search(r"SAID OBJECT: (.*)", user).group(1))
            return CallResult(
                {
                    "class": "search engines",
                    "said_canonical": said,
                    "gold": "AltaVista",
                    "compatible": [],
                },
                Usage(usd=0.01, calls=1),
            )
        if system in (K.DEDUPE_SYSTEM, K.DEDUPE_INCR_SYSTEM):
            return CallResult({"mapping": []}, Usage(calls=1))
        if system == K.CONCEPTS_SYSTEM:
            return CallResult({"members": [f"Engine{i:02d}" for i in range(30)]}, Usage(calls=1))
        if system == T.GEN_SYSTEM:
            cls = re.search(r"CLASS: (.*)", user).group(1)
            m = re.search(r"CONCEPT: (.*)", user)
            concept = m.group(1) if m else None
            kind = re.search(r"KIND: (\w+)", user).group(1)
            n = int(re.search(r"NUMBER OF TEXTS: (\d+)", user).group(1))
            texts = []
            for j in range(n):
                self.n += 1
                tag = f"zq{self.n}x{j}"
                if kind == "explicit":
                    a = f"Sure! {concept} is handy, right? Tag {tag} (really).\n\n- **Speed:** ok."
                    mention = concept
                else:
                    a = f"Sure! The blue-logo service {tag} is handy, right?\n\n- **Speed:** quick."
                    mention = "The blue-logo service" if kind == "implied" else ""
                texts.append(
                    {
                        "system": f"You are helpful ({cls}).",
                        "user": f"Tell me about item {tag}?",
                        "assistant": a,
                        "mention": mention,
                    }
                )
            return CallResult({"texts": texts}, Usage(usd=0.001, calls=1))
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    async def aclose(self):
        pass


class FakeTokenizer:
    def apply_chat_template(self, msgs, tokenize=False, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs) + "<|im_end|>\n"

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\w+|[^\w\s]+|\s+", text)]}


class Proc:
    def __init__(self, rc=0):
        self.rc = rc

    def poll(self):
        return self.rc


class FakeModal(R.ModalOps):
    """Plays Modal: generator runs land on a fake gen volume, capture / fit / score write
    what probes_modal.py would."""

    def __init__(self, out: Path, hang_main: bool = False):
        super().__init__("modal", {})
        self.out = out
        self.vol: dict[str, int] = {}  # run -> samples per key
        self.cmds: list[str] = []
        self.hang_main = hang_main  # main (non-precheck) generator runs never finish

    def volume(self, args):
        self.n_volume_ops += 1
        return 0, ""

    def ls(self, volume, path="/"):
        return list(self.vol)

    def apps(self):  # `modal app list --json` (none running unless a test says so)
        return list(getattr(self, "app_list", []))

    def start(self, cmd, cwd, log):
        self.n_launches += 1
        self.cmds.append(cmd)
        run = "gen-" + re.search(r"--run (\S+)", cmd).group(1)
        m = re.search(r"--(?:k|n-samples) (\d+)", cmd)
        self.vol[run] = int(m.group(1)) if m else 1
        return Proc(None if self.hang_main and "precheck" not in run else 0)

    def get(self, volume, remote, local):
        run = remote.strip("/")
        req = next(self.out.glob(f"resample/*/launch/{run}/request.json"))
        cells = [tuple(c) for c in json.loads(req.read_text())["cells"]]
        for i, la, p in cells:
            d = Path(local) / run / i
            d.mkdir(parents=True, exist_ok=True)
            with (d / f"L{la:03d}.jsonl").open("a") as fh:
                samples = [BLOCK.format(s="Google", j=j) for j in range(self.vol[run])]
                fh.write(json.dumps({"label": i, "layer": la, "pos": p, "samples": samples}) + "\n")
        return True

    def run(self, cmd, cwd=None):
        self.n_launches += 1
        self.cmds.append(cmd)
        if "--phase fit" in cmd:
            inp = json.loads(Path(re.search(r"--inputs (\S+)", cmd).group(1)).read_text())
            od = Path(re.search(r"--out-dir (\S+)", cmd).group(1))
            for slug, c in inp["classes"].items():
                assert c["probe_acts"], "per-class shard list"
                d = od / "fit" / slug
                d.mkdir(parents=True, exist_ok=True)
                nc = len(c["concepts"])
                for la in inp["layers"]:
                    res = {
                        "layer": la,
                        "class": c["class"],
                        "validity": {"a": [1.0] * nc},
                        "null_check": {"keys": {str(k): {"ok": True} for k in range(14)}},
                        "merge": {str(k): v for k, v in enumerate(c["merge"])},
                        "concepts": c["concepts"],
                    }
                    (d / f"fit_L{la:02d}.json").write_text(json.dumps(res))
                cen = {x: [float(i == j) for j in range(nc)] for i, x in enumerate(c["concepts"])}
                (d / "centroids_L44.json").write_text(json.dumps(cen))
        elif "--phase score" in cmd:
            inp = json.loads(Path(re.search(r"--inputs (\S+)", cmd).group(1)).read_text())
            od = Path(re.search(r"--out-dir (\S+)", cmd).group(1))
            for slug in inp["classes"]:
                fit = json.loads((od / "fit" / slug / "fit_L44.json").read_text())
                names = fit["concepts"]
                nc = len(names)
                d = od / "score" / slug
                d.mkdir(parents=True, exist_ok=True)
                its, pos = [], []
                for it in inp["items"]:
                    for p in inp["site_keys"][it]:
                        its.append(it)
                        pos.append(int(p))
                rng = np.random.default_rng(0)
                for la in inp["layers"]:
                    # the gold reads everywhere, the said objects never fire: rung 2 reaches
                    # cross_method / probe_unconfirmed
                    za = rng.normal(size=(len(its), nc)).astype(np.float32)
                    za[:, names.index("AltaVista")] = 5.0
                    for x in SAIDS:
                        if x in names:
                            za[:, names.index(x)] = -5.0
                    np.savez(
                        d / f"L{la:02d}.npz",
                        item=np.array(its),
                        pos=np.array(pos),
                        za=za,
                        key=np.zeros(len(its), int),
                    )
        return 0, ""


@pytest.fixture
def world(tmp_path, monkeypatch):
    s = sites()
    i0, i1 = ITEMS
    cells_a = [
        cell(i0, 44, s[i0][0], ["Google", "Bing"]),
        cell(i0, 48, s[i0][1], ["Yahoo"]),
        cell(i1, 44, s[i1][0], ["Baidu", "Google"]),
    ]
    cells_b = [cell(i1, 44, s[i1][2], ["Bing"])]
    cells_c = [cell(i0, 44, s[i0][3], ["Google"]), cell(i1, 48, s[i1][1], ["Yandex"])]
    root = tmp_path / "cells"
    write_cells(root / "lens-a" / "cells.jsonl", cells_a)
    write_cells(root / "tok-b" / "cells.jsonl", cells_b)
    write_cells(root / "lens-c" / "cells.jsonl", cells_c)
    snap = tmp_path / "snap"
    for n in ("gw-mlnla", "gw-wrong-bridge"):
        (snap / n).mkdir(parents=True)
        (snap / n / "f.py").write_text("x = 1\n")
    man = {
        "model_id": "m",
        "layers": L11,
        "prompts": [
            {
                "label": i,
                "family": "f",
                "file": f"/acts/acts-x/{i}.st",
                "n_pos": 200,
                "tokens": [],
                "targets": [],
            }
            for i in ITEMS
        ],
    }
    (tmp_path / "man.json").write_text(json.dumps(man))
    orig = tmp_path / "orig.jsonl"
    orig.write_text(
        "".join(
            json.dumps(
                {
                    "id": i,
                    "layer": la,
                    "pos": p,
                    "samples": [BLOCK.format(s="X", j=j) for j in range(3)],
                }
            )
            + "\n"
            for i in ITEMS
            for la in (44, 48)
            for p in s[i]
        )
    )
    pool = {
        "generator": "nla",
        "args": {"lens": "l", "acts": R.ACTS_11},
        "sampling": dict(R.SAMPLING),
        "orig_arm": "lens-a",
        "orig_readouts": str(orig),
        "gpu": "H200",
        "samples_per_run": 10,
        "usd_per_generated_sample": 0.001,
    }
    reg = {
        "gated_arms": ["lens-a", "tok-b"],
        "arms": {
            "lens-a": {
                "cells": None,
                "kind": "sampled",
                "family": "lens",
                "layers": L11,
                "pool": "pa",
            },
            "tok-b": {
                "cells": None,
                "kind": "token",
                "family": "tok-b",
                "layers": L11,
                "pool": None,
            },
            "lens-c": {
                "cells": None,
                "kind": "sampled",
                "family": "other",
                "layers": L11,
                "pool": "pc",
            },
        },
        # pc plays patchscopes: one seeded run per missing sample
        "pools": {
            "pa": dict(pool),
            "pc": {
                **pool,
                "generator": "lensagree",
                "args": {"acts": R.ACTS_11},
                "samples_per_run": 1,
                "orig_arm": "lens-c",
            },
        },
        "acts_manifests": {R.ACTS_11: str(tmp_path / "man.json")},
        "env": {
            "modal": "modal",
            "snapshot_root": str(snap),
            "acts_volume": "acts",
            "WSBENCH_GEN_VOLUME": "gen",
            "probe_run": "run",
            "tokenizer": "t",
        },
        "snapshots": {
            n: {
                "commit": "0" * 40,
                "source": "",
                "patches": [],
                "path": str(snap / n),
                "tree_sha256": G.tree_sha256(snap / n),
            }
            for n in ("gw-mlnla", "gw-wrong-bridge")
        },
    }
    reg_path = tmp_path / "arms.json"
    A.save(reg, reg_path)
    w = {
        "tmp": tmp_path,
        "reg_path": reg_path,
        "out": tmp_path / "outputs" / "sibling_gate",
        "data": tmp_path / "data",
        "cells": root,
    }
    return w


def make_gate(w, llm=None, modal=None, **kw):
    reg = A.load(w["reg_path"])
    added = kw.pop("added", [])
    if added:
        reg = A.with_added(reg, added)
    return G.Gate(
        reg=reg,
        reg_path=w["reg_path"],
        out=w["out"],
        data=w["data"],
        cells_root=w["cells"],
        llm=llm if llm is not None else FakeLLM(),
        modal=modal if modal is not None else FakeModal(w["out"]),
        tokenizer=FakeTokenizer(),
        quiet=True,
        yes=True,
        added=added,
        **kw,
    )


def labels(w):
    return {r["unit"]: r for f in (w["out"] / "labels").glob("*.jsonl") for r in U.iter_jsonl(f)}


@pytest.fixture
def gated(world):
    llm, modal = FakeLLM(), FakeModal(world["out"])
    g = make_gate(world, llm, modal)
    assert g.run() == G.EXIT_OK
    world["first"] = {"llm": llm, "modal": modal, "labels": labels(world)}
    return world


def test_end_to_end_gate_writes_labels_and_stamps(gated):
    w = gated
    lab = w["first"]["labels"]
    assert len(lab) == 6
    assert {r["rung1"] for r in lab.values()} == {"stable", "deterministic"}
    fin = {
        k.split(":", 1)[0] + ":" + k.split(":")[-2] + ":" + k.split("__")[0][-4:]: r["final"]
        for k, r in lab.items()
    }
    assert set(fin.values()) <= {"cross_method", "probe_unconfirmed"}
    assert fin["lens-a:google:0000"] == "probe_unconfirmed"  # nobody else names it yet
    gen = json.loads((w["out"] / "generation.json").read_text())
    assert gen["generation"] == 0 and gen["stamp"]["gated_arms"] == ["lens-a", "tok-b"]
    st = U.load_stamp("lens-a", w["out"] / "labels")
    assert st["generation"] == 0 and len(st["unit_hash"]) == 64
    assert w["first"]["modal"].n_launches > 0 and w["first"]["llm"].n > 0
    assert (w["out"] / "resample" / "pa" / "precheck.json").exists()
    assert (w["out"] / "changes.md").exists()


# ---------------------------------------------------------------- 3. idempotence


def test_plan_on_unchanged_inputs_is_zero_work_and_the_rerun_makes_no_calls(gated):
    w = gated
    g = make_gate(w)
    p = g.plan()
    assert p["zero_work"], p
    assert (
        p["flip_sources"]["new_arms"] == [] and p["flip_sources"]["cross_method_touched_units"] == 0
    )
    assert p["estimate"]["total_usd"] == 0
    llm, modal = FakeLLM(), FakeModal(w["out"])
    g = make_gate(w, llm, modal)
    assert g.run() == G.EXIT_OK
    assert llm.n == 0 and modal.n_launches == 0 and modal.n_volume_ops == 0
    assert labels(w) == w["first"]["labels"]
    assert json.loads((w["out"] / "generation.json").read_text())["generation"] == 0


# ---------------------------------------------------------------- 4-7. a new arm


def test_new_arm_incremental_bank_class_map_merge_and_distractors(gated):
    w = gated
    data, out = w["data"], w["out"]
    cm0 = json.loads((data / "class_map.json").read_text())
    slug = L.slug("search engines")
    bank0 = T.load_bank_class(data, slug)
    concepts0 = json.loads((data / "concepts.json").read_text())["classes"]["search engines"]
    merge0 = (out / "probes" / "merge.json").read_text()
    dis0 = json.loads((data / "distractors" / f"{slug}.json").read_text())
    # the (said, item) pairs lens-c adds: (google, i0) exists, (yandex, i1) is new
    p = make_gate(w, added=["lens-c"]).plan()
    assert not p["zero_work"] and p["resample"]["pc"]["needs_precheck"]
    assert p["flip_sources"]["new_arms"] == ["lens-c"]
    assert p["flip_sources"]["cross_method_touched_units"] >= 1
    assert A.load(w["reg_path"])["gated_arms"] == ["lens-a", "tok-b"]  # --plan changes nothing
    llm = FakeLLM()
    g = make_gate(w, llm, added=["lens-c"])
    assert g.run() == G.EXIT_OK
    assert A.load(w["reg_path"])["gated_arms"] == ["lens-a", "tok-b", "lens-c"]
    # 5. dedupe: old mappings unchanged; the new raw names went to one dedupe-incr call at most
    cm1 = json.loads((data / "class_map.json").read_text())
    for k, v in cm0["pairs"].items():
        assert cm1["pairs"][k] == v
    for k, v in cm0["names"].items():
        assert cm1["names"][k] == v
    assert sum(1 for c in llm.calls if c == K.DEDUPE_INCR_SYSTEM[:40]) <= 1
    assert sum(1 for c in llm.calls if c == K.DEDUPE_SYSTEM[:40]) == 0
    # 4. bank: texts only for the new concept, old texts untouched, fillers frozen
    c1 = json.loads((data / "concepts.json").read_text())["classes"]["search engines"]
    assert c1["fillers"] == concepts0["fillers"]
    assert c1["concepts"][: len(concepts0["concepts"])] == concepts0["concepts"]
    assert c1["concepts"][len(concepts0["concepts"]) :] == ["Yandex"]
    bank1 = T.load_bank_class(data, slug)
    assert bank1[: len(bank0)] == bank0
    assert {(t["concept"], t["kind"]) for t in bank1[len(bank0) :]} <= {
        ("Yandex", "explicit"),
        ("Yandex", "implied"),
    }
    assert all(t["avoid"] == c1["concepts"] for t in bank1[len(bank0) :])
    # 6. merge frozen (no --remerge)
    assert (out / "probes" / "merge.json").read_text() == merge0
    # 7. distractors: an existing unit's set is unchanged after the new arm adds a sibling name
    dis1 = json.loads((data / "distractors" / f"{slug}.json").read_text())
    for k, v in dis0.items():
        assert dis1[k] == v
    assert set(dis1) > set(dis0)
    assert len(bank1) > len(bank0)
    # lens-c (another family) names Google for item 0 at L44: lens-a's unit flips; the class
    # was refitted (new concept Yandex), so changes.md attributes it to "class" (first match)
    lab1 = labels(w)
    k = next(
        k for k in lab1 if k.startswith("lens-a:chat-dailydialog-0000__L044") and ":google:" in k
    )
    assert lab1[k]["final"] == "cross_method" and lab1[k]["rung2"]["cross_arms"] == ["lens-c"]
    assert w["first"]["labels"][k]["final"] == "probe_unconfirmed"
    ch = (out / "changes.md").read_text()
    assert "| lens-a | 1 | 1 | 0 | 0 | 0 |" in ch
    gen = json.loads((out / "generation.json").read_text())
    assert gen["generation"] == 1 and gen["changes"]["lens-a"] == {"class": 1}
    meta = json.loads((out / "probes" / "score" / slug / "meta.json").read_text())
    assert meta["units"] == {k: v["sha"] for k, v in dis1.items()}
    assert "lens-c" in labels(w)[next(k for k in dis1 if k.startswith("lens-c"))]["arm"]


def test_bank_cache_keys_of_existing_texts_untouched(gated):
    """Adding a concept requests texts for that concept only (sibgen-chat-v2.1 cells)."""
    w = gated
    idx = T.load_cells_index(w["data"])
    row = json.loads((w["data"] / "concepts.json").read_text())["classes"]["search engines"]
    assert T.missing_cells("search engines", row["concepts"], idx) == []
    assert T.missing_cells("search engines", [*row["concepts"], "Lycos"], idx) == [
        ("Lycos", "explicit"),
        ("Lycos", "implied"),
    ]


def test_dedupe_incr_one_call_and_no_call_without_new_names(tmp_path):
    llm = FakeLLM()

    async def go(new):
        with Cache(tmp_path / "c.jsonl") as cache:
            return await K.run_dedupe_incr(new, ["search engines"], llm, cache)

    assert asyncio.run(go([])) == ({}, True) and llm.n == 0
    mp, frozen = asyncio.run(go(["web search engines", "browsers"]))
    assert frozen and mp == {"web search engines": "web search engines", "browsers": "browsers"}
    assert llm.n == 1
    asyncio.run(go(["web search engines", "browsers"]))  # cached
    assert llm.n == 1


def test_merge_frozen_unless_remerge(gated):
    w = gated
    merge = json.loads((w["out"] / "probes" / "merge.json").read_text())
    merge["search engines"] = [13] * 14  # pretend an old, different merge
    (w["out"] / "probes" / "merge.json").write_text(json.dumps(merge))
    g = make_gate(w)
    assert g.capture_plan(json.loads((w["data"] / "concepts.json").read_text())["classes"]) == {}
    g = make_gate(w, remerge=["search engines"])
    cp = g.capture_plan(json.loads((w["data"] / "concepts.json").read_text())["classes"])
    assert cp["search engines"] == len(T.load_bank_class(w["data"], L.slug("search engines")))


# ---------------------------------------------------------------- 8. the cells guard


def test_unit_set_change_regates_and_mismatched_labels_fall_back(gated, tmp_path):
    w = gated
    s = sites()
    write_cells(
        w["cells"] / "lens-a" / "cells.jsonl", [cell(ITEMS[0], 44, s[ITEMS[0]][0], ["Google"])]
    )
    p = make_gate(w).plan()
    assert p["arms"]["lens-a"]["changed"] and not p["arms"]["tok-b"]["changed"]
    ld = w["out"] / "labels"
    new_hash = U.unit_set_hash(U.build_units("lens-a", w["cells"] / "lens-a" / "cells.jsonl")[0])
    assert U.labels_guard("lens-a", ld, new_hash) == "mismatch"
    assert (
        U.labels_guard("tok-b", ld, json.loads((ld / "tok-b.gen.json").read_text())["unit_hash"])
        == "ok"
    )
    assert U.labels_guard("lens-a", tmp_path / "none", new_hash) == "no_labels"
    # report consumers refuse labels the rebuild refused
    d = tmp_path / "lens-a"
    d.mkdir()
    (d / "summary_v2.json").write_text(json.dumps({"config": {"sibling_labels_guard": "mismatch"}}))
    assert report_v2._labels(d, ld) is None
    (d / "summary_v2.json").write_text(json.dumps({"config": {"sibling_labels_guard": "ok"}}))
    assert report_v2._labels(d, ld)
    # the re-run re-gates the arm (new unit hash in the stamp)
    assert make_gate(w).run() == G.EXIT_OK
    assert U.labels_guard("lens-a", ld, new_hash) == "ok"


def test_rebuild_refuses_mismatched_labels_and_scores_rev20(tmp_path, prose_readouts):
    from test_claimmap_v2 import MapFake, _run

    from hallucination_bench.evals.wsbench_chat import run as wrun

    out = tmp_path / "o"
    _run([("p", prose_readouts)], out, MapFake(), stages=["map"])
    arm_dir = wrun.arm_dir(out, wrun.Models(), "p")
    s20 = json.loads((arm_dir / "summary_v2.json").read_text())
    ld = tmp_path / "labels"
    ld.mkdir()
    (ld / "p.jsonl").write_text(
        json.dumps({"unit": "u", "arm": "p", "final": "unstable", "segments": []}) + "\n"
    )
    for h, guard in (("0" * 64, "mismatch"), (None, "no_stamp")):
        (ld / "p.gen.json").write_text(json.dumps({"generation": 3, "unit_hash": h}))
        wrun.rebuild_arm("p", prose_readouts, out, wrun.Models(), ld)
        s = json.loads((arm_dir / "summary_v2.json").read_text())
        assert s["config"]["sibling_labels_guard"] == guard
        assert (s["config"]["sibling_labels"] is None) == (guard == "mismatch")
        assert s["overall"] == s20["overall"]  # the toy labels join nothing either way


# ---------------------------------------------------------------- 9. report


def test_report_reads_labels_by_default_and_no_sibling_labels_gives_rev20(gated, tmp_path):
    w = gated
    ld = w["out"] / "labels"
    md = "\n".join(report_v2.gate_section(ld, ["lens-a", "tok-b", "other"], {}))
    assert "Gate generation **0**" in md
    assert "| lens-a | gated |" in md and "ungated (not in the gate set)" in md
    off = "\n".join(report_v2.gate_section(U.NO_LABELS, ["lens-a"], {}))
    assert "--no-sibling-labels" in off
    assert U.load_labels("lens-a", U.NO_LABELS) is None
    assert U.labels_dir(tmp_path) == tmp_path / "sibling_gate" / "labels"
    from hallucination_bench import cli

    a = cli.build_parser().parse_args(["report", "--view", "v2", "--no-sibling-labels"])
    assert a.no_sibling_labels
    a = cli.build_parser().parse_args(["report", "--view", "v2"])
    assert not a.no_sibling_labels and a.sibling_labels is None  # default: <out>/sibling_gate


def test_needs_yes_over_the_cap_exits_2(world):
    g = make_gate(world, auto_usd=0.0)
    g.yes = False
    assert g.run() == G.EXIT_NEEDS_YES
    assert not (world["out"] / "labels").exists()


def test_diff_labels_parsed_rows(gated, tmp_path):
    w = gated
    base = tmp_path / "base"
    base.mkdir()
    for f in (w["out"] / "labels").glob("*.jsonl"):
        (base / f.name).write_text(f.read_text())
    assert G.diff_labels(w["out"] / "labels", base, ["lens-a", "tok-b"])["n_diff"] == 0
    rows = [json.loads(x) for x in (base / "lens-a.jsonl").read_text().splitlines()]
    rows[0]["final"] = "unstable"
    (base / "lens-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    d = G.diff_labels(w["out"] / "labels", base, ["lens-a", "tok-b"])
    assert d["n_diff"] == 1 and d["per_arm"]["lens-a"]["fields"] == {"final": 1}


def test_retry_failed_contextvar_keeps_cached_failures(tmp_path):
    calls = []

    class Bad:
        async def call(self, **kw):
            calls.append(1)
            return CallResult(None, Usage())

    async def go():
        with Cache(tmp_path / "c.jsonl") as cache:
            return await F.cached_json_call(
                cache,
                Bad(),
                key="k",
                version="v",
                model="m",
                effort=None,
                system="s",
                user="u",
                schema={},
            )

    assert asyncio.run(go())[0] == "failed" and len(calls) == 1
    tok = F.RETRY_FAILED.set(False)
    try:
        assert asyncio.run(go())[0] == "failed" and len(calls) == 1  # no re-ask
    finally:
        F.RETRY_FAILED.reset(tok)
    assert asyncio.run(go())[0] == "failed" and len(calls) == 2  # default: re-asked


def test_seeded_pool_and_resample_timeout_resume(world):
    w = world
    w_reg = A.load(w["reg_path"])
    w_reg["gated_arms"] = ["lens-c"]
    A.save(w_reg, w["reg_path"])
    modal = FakeModal(w["out"], hang_main=True)
    g = make_gate(w, FakeLLM(), modal, resample_timeout_s=0)
    with pytest.raises(G.GateExit) as e:
        g.run()
    assert e.value.code == G.EXIT_PARTIAL
    st = R.load_state("pc", w["out"])
    launched = [r for r, v in st.runs.items() if v["status"] == "launched"]
    assert len(launched) == 10 and all(r.endswith(f"-r{j}") for j, r in enumerate(launched))
    assert len({st.runs[r]["seed"] for r in launched}) == 10
    seeds = [int(re.search(r"--seed (\d+)", c).group(1)) for c in modal.cmds if "precheck" not in c]
    assert sorted(seeds) == sorted(st.runs[r]["seed"] for r in launched)
    # the re-run lists the volume, collects the 10 seeded runs and launches nothing new
    modal2 = FakeModal(w["out"])
    modal2.vol = dict(modal.vol)
    g = make_gate(w, FakeLLM(), modal2)
    assert g.run() == G.EXIT_OK
    assert modal2.n_launches == 0  # no relaunch (2 pairs: below the class cap, no probes)
    complete, counts = R.merge_runs(R.load_runs("pc", w["out"]))
    assert set(counts.values()) == {10} and len(complete) == 2


def test_snapshot_drift_is_refused(world, tmp_path):
    reg = A.load(world["reg_path"])
    d = G.snapshot_dir(reg, "gw-mlnla")
    assert G.check_snapshot(reg, "gw-mlnla", False, print) == d
    (d / "f.py").write_text("x = 2\n")
    with pytest.raises(G.GateExit) as e:
        G.check_snapshot(reg, "gw-mlnla", False, print)
    assert e.value.code == G.EXIT_FATAL
    msgs = []
    G.check_snapshot(reg, "gw-mlnla", True, msgs.append)
    assert "drift allowed" in msgs[0]


def test_lazy_llm_survives_multiple_event_loops():
    import asyncio

    from hallucination_bench.evals.sibling_gate.gate import LazyLLM

    class Fake:
        def __init__(self):
            self.sem = asyncio.Semaphore(2)

        async def call(self, **kw):
            async with self.sem:
                return {"ok": True}

    built = []

    def factory():
        built.append(1)
        return Fake()

    llm = LazyLLM(rpm=10, concurrency=2, factory=factory)
    for _ in range(2):  # two asyncio.run stages, as in the gate (fill, then classes)
        assert asyncio.run(llm.call(model="m")) == {"ok": True}
    assert len(built) == 2 and llm.n_calls == 2


def test_tree_sha_ignores_pycache(tmp_path):
    from hallucination_bench.evals.sibling_gate.gate import tree_sha256

    (tmp_path / "a.py").write_text("x = 1\n")
    h = tree_sha256(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0\1")
    assert tree_sha256(tmp_path) == h


# ---------------------------------------------------------------- review fixes (plan 0010)


def gen_cmds(modal, pool="pa"):
    return [c for c in modal.cmds if f"--run sibgate-{pool}-" in c and "precheck" not in c]


def test_precheck_failed_on_disk_never_launches_full_runs(world):
    w = world
    d = w["out"] / "resample" / "pa"
    d.mkdir(parents=True)
    (d / "precheck.json").write_text(json.dumps({"ok": False, "keys": []}))
    modal = FakeModal(w["out"])
    g = make_gate(w, FakeLLM(), modal)
    assert g.run() == G.EXIT_OK
    assert "pa" in g.failed_pools and gen_cmds(modal) == []
    assert not any("precheck" in c for c in modal.cmds)  # not re-prechecked either


class FlakyModal(FakeModal):
    """Main generator runs fail in one way: ``mode`` = rc (exit 1, 3 samples), get (the volume
    get fails), empty (the run dir has no rows)."""

    def __init__(self, out, mode):
        super().__init__(out)
        self.mode = mode

    def start(self, cmd, cwd, log):
        p = super().start(cmd, cwd, log)
        run = "gen-" + re.search(r"--run (\S+)", cmd).group(1)
        if "precheck" in run:
            return p
        if self.mode == "rc":
            self.vol[run] = 3
            return Proc(1)
        return p

    def get(self, volume, remote, local):
        if "precheck" in remote:
            return super().get(volume, remote, local)
        if self.mode == "get":
            return False
        if self.mode == "empty":
            (Path(local) / remote.strip("/")).mkdir(parents=True, exist_ok=True)
            return True
        return super().get(volume, remote, local)


@pytest.mark.parametrize("mode", ["rc", "get", "empty"])
def test_failed_or_empty_runs_do_not_consume_attempts(world, mode):
    w = world
    modal = FlakyModal(w["out"], mode)
    g = make_gate(w, FakeLLM(), modal)
    with pytest.raises(G.GateExit) as e:
        g.run()
    assert e.value.code == G.EXIT_PARTIAL  # never filled as resample_failed
    assert not (w["out"] / "labels").exists()
    st = R.load_state("pa", w["out"])
    keys = [
        tuple(k)
        for k in json.loads(next(w["out"].glob("resample/pa/launch/*/keys.json")).read_text())
    ]
    assert len(gen_cmds(modal)) == R.MAX_ATTEMPTS
    assert all(st.n_attempts(k) == 0 and st.n_launched(k) == 2 for k in keys)
    assert {v["status"] for r, v in st.runs.items()} == {"failed"}
    # distinct run names per launch (a1, a2), and a healthy re-run completes
    assert len({re.search(r"--run (\S+)", c).group(1) for c in gen_cmds(modal)}) == 2
    modal2 = FakeModal(w["out"])
    assert make_gate(w, FakeLLM(), modal2).run() == G.EXIT_OK
    assert len(gen_cmds(modal2)) == 1


def test_short_rows_consume_attempts_then_resample_failed(world):
    """Runs that return rows (short) DO count: after 2 attempts the key is exhausted."""
    w = world

    class Short(FakeModal):
        def start(self, cmd, cwd, log):
            p = super().start(cmd, cwd, log)
            run = "gen-" + re.search(r"--run (\S+)", cmd).group(1)
            if "precheck" not in run:
                self.vol[run] = 3
            return p

    modal = Short(w["out"])
    assert make_gate(w, FakeLLM(), modal).run() == G.EXIT_OK
    st = R.load_state("pa", w["out"])
    assert len(gen_cmds(modal)) == 2 and set(st.attempts.values()) == {2}
    assert {r["rung1"] for r in labels(w).values() if r["arm"] == "lens-a"} == {"resample_failed"}


def test_timeout_then_rerun_waits_for_in_flight_runs(world, monkeypatch):
    w = world
    monkeypatch.setattr(G, "POLL_S", 0)
    modal = FakeModal(w["out"], hang_main=True)
    with pytest.raises(G.GateExit) as e:
        make_gate(w, FakeLLM(), modal, resample_timeout_s=0).run()
    assert e.value.code == G.EXIT_PARTIAL
    (run,) = [r for r, v in R.load_state("pa", w["out"]).runs.items() if v["status"] == "launched"]
    finished = dict(modal.vol)

    class Running(FakeModal):
        n_polls = 0

        def apps(self):
            self.n_polls += 1
            if self.n_polls < 3:  # still running: nothing on the volume yet
                return [{"id": "ap-x", "description": run, "state": "ephemeral (detached)"}]
            self.vol.update(finished)  # the app finished writing
            return [{"id": "ap-x", "description": run, "state": "stopped"}]

    modal2 = Running(w["out"])
    assert make_gate(w, FakeLLM(), modal2).run() == G.EXIT_OK
    assert modal2.n_polls >= 3 and gen_cmds(modal2) == []  # waited, never relaunched
    assert R.load_state("pa", w["out"]).runs[run]["status"] == "collected"


def test_in_flight_by_volume_mtime_when_no_app_is_tied(world):
    w = world
    modal = FakeModal(w["out"], hang_main=True)
    with pytest.raises(G.GateExit):
        make_gate(w, FakeLLM(), modal, resample_timeout_s=0).run()
    (run,) = [r for r, v in R.load_state("pa", w["out"]).runs.items() if v["status"] == "launched"]

    class Fresh(FakeModal):
        def volume_mtimes(self, volume, path="/"):
            return {run: __import__("time").time() - 60}

    modal2 = Fresh(w["out"])
    with pytest.raises(G.GateExit) as e:
        make_gate(w, FakeLLM(), modal2, resample_timeout_s=0).run()
    assert e.value.code == G.EXIT_PARTIAL and "still running" in str(e.value)
    assert modal2.cmds == []  # no second attempt while the first may still be running


def test_modal_app_list_and_volume_ls_parsing():
    class M(R.ModalOps):
        def sh(self, cmd, cwd=None):
            if "app list" in cmd:
                return 0, json.dumps(
                    [{"App ID": "ap-1", "Description": "gen-x", "State": "ephemeral (detached)"}]
                )
            return 0, json.dumps(
                [{"Filename": "gen-x", "Created/Modified": "2026-10-16 10:00:00+00:00"}]
            )

    m = M("modal", {})
    assert m.apps() == [{"id": "ap-1", "description": "gen-x", "state": "ephemeral (detached)"}]
    assert R.app_running("ephemeral (detached)") and not R.app_running("stopped")
    assert m.volume_mtimes("gen")["gen-x"] > 0


def test_atomic_writes_keep_the_old_file_on_a_crash(tmp_path, monkeypatch):
    T.write_bank_class(tmp_path, "c", [{"id": "a"}])
    T.save_cells_index(tmp_path, {"c": [1]})
    p = tmp_path / "x.json"
    g = G.Gate.__new__(G.Gate)
    g.write_if_changed(p, {"v": 1})
    old = {
        f: f.read_bytes()
        for f in [T.bank_file(tmp_path, "c"), T.bank_dir(tmp_path) / "cells.json", p]
    }

    def boom(self, target):
        raise OSError("crash")

    monkeypatch.setattr(Path, "replace", boom)
    for fn in (
        lambda: T.write_bank_class(tmp_path, "c", [{"id": "b"}]),
        lambda: T.save_cells_index(tmp_path, {"c": [2]}),
        lambda: g.write_if_changed(p, {"v": 2}),
    ):
        with pytest.raises(OSError):
            fn()
    assert {f: f.read_bytes() for f in old} == old


def test_cells_index_saved_per_class_with_its_bank(gated, monkeypatch):
    """A crash at the 2nd class's bank write leaves the 1st class's bank AND index on disk."""
    w = gated
    g = make_gate(w)
    concepts = json.loads((w["data"] / "concepts.json").read_text())["classes"]
    c0 = sorted(concepts)[0]
    concepts = {c0: concepts[c0], "zz copy": dict(concepts[c0])}
    idx = T.load_cells_index(w["data"])
    idx.pop(c0)  # pretend c0's cells were never generated
    T.save_cells_index(w["data"], idx)
    real, n = T.write_bank_class, []

    def write(d, slug, rows):
        n.append(slug)
        if len(n) == 2:
            raise OSError("crash")
        return real(d, slug, rows)

    monkeypatch.setattr(T, "write_bank_class", write)
    units, _ = g.load_units()
    classes = json.loads((w["data"] / "classes.json").read_text())["classes"]
    with pytest.raises(OSError):
        g.run_texts(units, classes, concepts)
    first = c0 if n[0] == L.slug(c0) else "zz copy"
    assert first in T.load_cells_index(w["data"])


def test_stamp_changes_with_label_outcomes(gated):
    g = make_gate(gated)
    rows = list(labels(gated).values())
    a = g.stamp({}, {}, {}, rows)
    rows2 = [dict(r) for r in rows]
    rows2[0]["final"] = "probe_na" if rows2[0].get("final") != "probe_na" else "cross_method"
    assert G.sha256_json(a) != G.sha256_json(g.stamp({}, {}, {}, rows2))
    assert G.sha256_json(a) == G.sha256_json(g.stamp({}, {}, {}, list(reversed(rows))))


def test_modal_counters_are_thread_safe():
    import threading

    class M(R.ModalOps):
        def sh(self, cmd, cwd=None):
            return 0, ""

    m = M("modal", {})
    ts = [
        threading.Thread(target=lambda: [(m.run("x"), m.volume("y")) for _ in range(500)])
        for _ in range(8)
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert m.n_launches == 4000 and m.n_volume_ops == 4000


@pytest.mark.parametrize("val", ["abc", "nan", "inf"])
def test_non_numeric_gate_auto_usd_is_fatal(monkeypatch, val):
    monkeypatch.setenv("GATE_AUTO_USD", val)
    with pytest.raises(G.GateExit) as e:
        G.auto_usd_from_env()
    assert e.value.code == G.EXIT_FATAL and "GATE_AUTO_USD" in str(e.value)
    monkeypatch.setenv("GATE_AUTO_USD", "12.5")
    assert G.auto_usd_from_env() == 12.5
    monkeypatch.delenv("GATE_AUTO_USD")
    assert G.auto_usd_from_env() == G.GATE_AUTO_USD_DEFAULT


def test_gate_section_says_labels_refused(gated):
    ld = gated["out"] / "labels"
    md = "\n".join(
        report_v2.gate_section(
            ld, ["lens-a"], {"lens-a": {"config": {"sibling_labels_guard": "mismatch"}}}
        )
    )
    assert "labels refused (unit hash mismatch; re-run gate)" in md


def test_lifetime_launch_cap_stops_always_failing_pool():
    from hallucination_bench.evals.sibling_gate import resample as R

    st = R.PoolState("p")
    k = ("chat-x", 44, 7)
    for _ in range(R.MAX_LAUNCHES):  # every launch failed and was voided
        st.record_launch([k])
        st.voided[st.kstr(k)] = st.voided.get(st.kstr(k), 0) + 1
    assert st.n_attempts(k) == 0 and st.n_launched(k) == R.MAX_LAUNCHES
    assert st.spent(k)
    launches, failed = R.plan_launches("p", [k], {}, st, R.N_RESAMPLE)
    assert launches == [] and failed == [k]
