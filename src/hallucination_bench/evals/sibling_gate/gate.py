"""Plan 0010: ``hbench gate`` -- the sibling gate as a normal pipeline step.

One command gates every arm of the persisted gated-arm set (``data/arms.json`` ``gated_arms``)
after the claim map::

    hbench gate [--plan] [--yes] [--arms a,b] [--add-arm X] [--cells-root DIR]
    hbench gate import --from /workspace/hb-sibling-resample   # plan 0010 §10 (no calls)
    hbench gate snapshot                                       # §1 generator snapshots
    hbench gate diff --baseline DIR                            # parsed-row label diff

The gate is a **generation**: every run re-labels every gated arm (the cross-method index, the
shared-prior discount, the matched null, the class pair counts / >= 4-pair cap and
``item_sibling_names`` are computed over the whole set). The expensive work is incremental
through caches and frozen state: resample runs (§2), the fill / class / concept / text caches
(§4), the frozen class map, fillers, merge and per-unit distractor sets, the per-class fit input
hash and the per-class score ``meta.json`` (§5, §10). ``--plan`` prints the work and a cost
estimate and changes nothing. A real run proceeds without ``--yes`` only if the estimate is
<= ``$GATE_AUTO_USD`` (default $300).

Exit codes: 0 done; 2 needs ``--yes``; 3 fatal (config / auth / spend limit / unregistered
arm); 4 partial (re-run resumes).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hallucination_bench import DATA, REPO_ROOT
from hallucination_bench import arms as A

from . import resample as R
from . import units as U

DATA_DIR = DATA / "sibling_gate"
SCRIPTS = REPO_ROOT / "scripts" / "sibling_gate"
GATE_AUTO_USD_DEFAULT = 300.0
MIN_CLASS_PAIRS = 4  # rev 8.2: only classes with >= 4 (said, item) pairs get probes
CODE_FILES = ("probe_fit.py", "probes.py", "um_probe.py")
HEARTBEAT_S = 300
POLL_S = 30
INFLIGHT_FRESH_S = 30 * 60  # a run dir changed this recently (no app tied): still running

EXIT_OK, EXIT_NEEDS_YES, EXIT_FATAL, EXIT_PARTIAL = 0, 2, 3, 4
CAPTURE_PER_SHARD = 1500  # prompts per capture container (one ~= 200 prompts / min)
EXIT_MEANING = {
    EXIT_OK: "done",
    EXIT_NEEDS_YES: "needs --yes (estimate over $GATE_AUTO_USD)",
    EXIT_FATAL: "fatal: config / auth / spend limit / unregistered arm",
    EXIT_PARTIAL: "partial; re-run resumes",
}


class GateExit(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- small helpers


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(x: Any) -> str:
    return sha256_bytes(json.dumps(x, sort_keys=True, ensure_ascii=False).encode())


def sha256_file(p: Path) -> str | None:
    p = Path(p)
    return sha256_bytes(p.read_bytes()) if p.exists() else None


def code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {f: sha256_file(here / f) or "" for f in CODE_FILES}


def code_revision() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        rev = r.stdout.strip() or "unknown"
        d = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return rev + ("+dirty" if d.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_json(p: Path, x: Any, indent: int | None = 1) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=indent, ensure_ascii=False) + "\n")
    tmp.replace(p)


def read_json(p: Path, default: Any = None) -> Any:
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def items() -> dict[str, dict[str, Any]]:
    from hallucination_bench.evals.wsbench_chat import bank

    return {it["id"]: it for it in bank.load_items()}


def site_tokens(its: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, int], str]:
    return {
        (i, int(s["pos"])): s.get("token") or "" for i, row in its.items() for s in row["sites"]
    }


def eval_sites_hash(its: Mapping[str, Mapping[str, Any]]) -> str:
    """The score phase's eval-site set: every item's sites, their keys, the layers, the acts."""
    from . import ladder as L
    from . import texts as T

    keys = {
        it: {str(s["pos"]): T.token_class(s.get("token") or "") for s in its[it]["sites"]}
        for it in its
    }
    return sha256_json(
        {
            "items": sorted(its),
            "site_keys": keys,
            "layers": list(L.PROBE_LAYERS),
            "eval_acts": eval_acts(),
        }
    )


def eval_acts() -> dict[str, str]:
    from . import ladder as L

    return {str(la): (R.ACTS_L42 if la == 42 else R.ACTS_11) for la in L.PROBE_LAYERS}


def distractor_sha(rec: Mapping[str, Any]) -> str:
    return sha256_json({k: v for k, v in rec.items() if k not in ("sha", "generation")})


def fit_slice(
    cls: str, concepts_row: Mapping[str, Any], bank_rows: list[dict[str, Any]], merge: list[int]
) -> dict[str, Any]:
    """The class's slice of ``probe_inputs.json``, derived from the bank (``cmd_fit`` logic)."""
    from . import probes as P
    from .stages import probe_label

    ci = {c: i for i, c in enumerate(concepts_row["concepts"])}
    folds = P.assign_folds(bank_rows, salt=cls)
    return {
        "class": cls,
        "concepts": list(concepts_row["concepts"]),
        "merge": list(merge),
        "texts": [
            {
                "id": probe_label(t["id"]),
                "c": ci.get(t["concept"], P.NEUTRAL) if t["concept"] else P.NEUTRAL,
                "kind": t["kind"],
                "request_idx": 0,
                "fold": folds[t["id"]],
                "key": (t.get("capture") or {}).get("key"),
            }
            for t in bank_rows
        ],
    }


def class_shards(bank_rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({(t.get("capture") or {}).get("shard") for t in bank_rows} - {None})


def fit_input_hash(
    sl: Mapping[str, Any], bank_rows: Iterable[Mapping[str, Any]], shards: list[str]
) -> str:
    """§10.4: texts (sorted ids + content), folds, frozen merge, concepts, shard list, layers and
    the probe_fit / probes / um_probe file hashes."""
    from . import ladder as L

    content = sorted((t["id"], t["system"], t["user"], t["assistant"]) for t in bank_rows)
    return sha256_json(
        {
            "slice": sl,
            "content": content,
            "shards": sorted(shards),
            "layers": list(L.PROBE_LAYERS),
            "code": code_hashes(),
        }
    )


class LazyLLM:
    """Builds the real :class:`~hallucination_bench.llm.LLM` on the first call only (an
    unchanged gate never constructs a client) and counts calls."""

    def __init__(self, rpm: float, concurrency: int, factory=None) -> None:
        self.rpm, self.concurrency = rpm, concurrency
        self.factory = factory
        self._llm = None
        self.n_calls = 0

    async def call(self, **kw):
        # the gate runs several asyncio.run() stages; the client's semaphore / pacer bind to the
        # loop that built them, so rebuild the client when the running loop changes
        loop = asyncio.get_running_loop()
        if self._llm is not None and getattr(self, "_loop", None) is not loop:
            self._llm = None
        self._loop = loop
        if self._llm is None:
            if self.factory is not None:
                self._llm = self.factory()
            else:
                from hallucination_bench.llm import LLM

                self._llm = LLM(rpm=self.rpm, concurrency=self.concurrency)
        self.n_calls += 1
        return await self._llm.call(**kw)

    async def aclose(self) -> None:
        if self._llm is not None and hasattr(self._llm, "aclose"):
            await self._llm.aclose()


class OfflineLLM(LazyLLM):
    """``--offline``: any LLM call is a bug (the idempotence check, plan §8 step 1)."""

    async def call(self, **kw):
        raise GateExit(EXIT_FATAL, "--offline: the gate tried an LLM call")


class OfflineModal(R.ModalOps):
    """``--offline``: any Modal operation (launch or volume) is a bug."""

    def sh(self, cmd: str, cwd: Path | None = None):
        raise GateExit(EXIT_FATAL, f"--offline: the gate tried a Modal command: {cmd[:200]}")

    def start(self, cmd: str, cwd: Path, log: Path):
        raise GateExit(EXIT_FATAL, f"--offline: the gate tried a Modal launch: {cmd[:200]}")


# ---------------------------------------------------------------- snapshots (§1)


def tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    root = Path(root)
    files = (
        x
        for x in root.rglob("*")
        if x.is_file() and "__pycache__" not in x.parts and x.suffix != ".pyc"
    )  # Python writes bytecode into the snapshot while jobs run: never part of the tree
    for p in sorted(files):
        h.update(str(p.relative_to(root)).encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def snapshot_dir(reg: Mapping[str, Any], name: str) -> Path:
    s = reg["snapshots"][name]
    return Path(s.get("path") or Path(reg["env"]["snapshot_root"]) / f"{name}-{s['commit'][:8]}")


def make_snapshot(reg: dict[str, Any], name: str, dest_root: Path | None = None) -> dict[str, Any]:
    """``git archive`` the commit from the local clone (read-only), apply the allowed patches,
    record the tree checksum in the registry entry (returned; the caller saves)."""
    s = reg["snapshots"][name]
    src = Path(s["source"])
    ok = subprocess.run(
        ["git", "-C", str(src), "cat-file", "-e", f"{s['commit']}^{{commit}}"],
        capture_output=True,
    )
    if ok.returncode != 0:
        raise GateExit(EXIT_FATAL, f"snapshot {name}: commit {s['commit']} missing in {src}")
    dest = Path(dest_root) / f"{name}-{s['commit'][:8]}" if dest_root else snapshot_dir(reg, name)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    arch = subprocess.run(
        ["git", "-C", str(src), "archive", "--format=tar", s["commit"]], capture_output=True
    )
    if arch.returncode != 0:
        raise GateExit(EXIT_FATAL, f"snapshot {name}: git archive failed: {arch.stderr[-400:]!r}")
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(arch.stdout)) as tf:
        tf.extractall(dest, filter="data")
    for pt in s.get("patches") or []:
        f = dest / pt["file"]
        txt = f.read_text()
        if txt.count(pt["old"]) != 1:
            raise GateExit(EXIT_FATAL, f"snapshot {name}: patch target not unique in {pt['file']}")
        f.write_text(txt.replace(pt["old"], pt["new"]))
    s["path"] = str(dest)
    s["tree_sha256"] = tree_sha256(dest)
    return s


def check_snapshot(reg: Mapping[str, Any], name: str, allow_drift: bool, log) -> Path:
    s = reg["snapshots"][name]
    d = snapshot_dir(reg, name)
    if not s.get("tree_sha256") or not d.exists():
        raise GateExit(EXIT_FATAL, f"snapshot {name} missing: run `hbench gate snapshot` first")
    got = tree_sha256(d)
    if got != s["tree_sha256"]:
        if not allow_drift:
            raise GateExit(
                EXIT_FATAL,
                f"snapshot {name} drifted ({got[:12]} != {s['tree_sha256'][:12]}); "
                "re-run `hbench gate snapshot` or pass --allow-snapshot-drift",
            )
        log(f"WARNING: snapshot {name} drift allowed (--allow-snapshot-drift): {got[:12]}")
    return d


# ---------------------------------------------------------------- the gate


@dataclass
class Gate:
    reg: dict[str, Any]
    reg_path: Path = A.REGISTRY_PATH
    out: Path = U.OUT
    data: Path = DATA_DIR
    cells_root: Path = U.CELLS_ROOT_DEFAULT
    plan_only: bool = False
    yes: bool = False
    show_arms: list[str] | None = None
    added: list[str] = field(default_factory=list)
    rung1_skip: bool = False
    retry_failed: bool = False
    remerge: list[str] = field(default_factory=list)
    allow_snapshot_drift: bool = False
    capture_shards: int = 0  # 0 = ceil(n / CAPTURE_PER_SHARD)
    resample_timeout_s: float = 10 * 3600
    auto_usd: float = GATE_AUTO_USD_DEFAULT
    rpm: float = 1500.0
    concurrency: int = 64
    llm: Any = None
    modal: Any = None
    tokenizer: Any = None
    quiet: bool = False
    offline: bool = False
    log_lines: list[str] = field(default_factory=list)
    work: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out, self.data = Path(self.out), Path(self.data)
        self.probes = self.out / "probes"
        env = {k: v for k, v in self.reg["env"].items() if k.isupper()}
        if self.llm is None:
            self.llm = OfflineLLM(0, 1) if self.offline else LazyLLM(self.rpm, self.concurrency)
        elif not isinstance(self.llm, LazyLLM):  # an injected client: count its calls
            inner = self.llm
            self.llm = LazyLLM(self.rpm, self.concurrency, factory=lambda: inner)
        if self.modal is None:
            mk = OfflineModal if self.offline else R.ModalOps
            self.modal = mk(self.reg["env"]["modal"], env)
        self.gated = list(self.reg["gated_arms"])
        self.its = None
        self.spend_usd = 0.0
        self.deterministic = set(A.deterministic_arms(self.reg))
        self.skipped_rung1: list[str] = []
        self.failed_pools: set[str] = set()
        self.seeded: set[str] = set()

    # ------------------------------------------------------------ logging
    def log(self, msg: str) -> None:
        line = f"[gate {time.strftime('%H:%M:%S')}] {msg}"
        self.log_lines.append(line)
        if not self.quiet:
            print(line, flush=True)

    def items(self) -> dict[str, dict[str, Any]]:
        if self.its is None:
            self.its = items()
        return self.its

    # ------------------------------------------------------------ registry checks
    def check_registry(self) -> None:
        for a in self.gated:
            row = self.reg["arms"].get(a)
            if row is None:
                raise GateExit(EXIT_FATAL, f"arm {a} is not registered in data/arms.json")
            if row["kind"] == "sampled" and not row.get("pool"):
                if not self.rung1_skip:
                    raise GateExit(
                        EXIT_FATAL,
                        f"sampled arm {a} has no registered pool (register one in "
                        "data/arms.json, or --rung1 skip to gate it as deterministic)",
                    )
                self.skipped_rung1.append(a)
                self.deterministic.add(a)
                self.log(f"WARNING: --rung1 skip: {a} has no pool; its units are deterministic")

    # ------------------------------------------------------------ §3 units
    def load_units(self) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Every gated arm's units (gated order) from its registered cells (cached by the cells
        file's size + mtime), and the per-arm unit hash."""
        allu: list[dict[str, Any]] = []
        hashes: dict[str, str] = {}
        for a in self.gated:
            p = A.cells_path(a, self.cells_root, self.reg)
            if not p.exists():
                raise GateExit(EXIT_FATAL, f"[{a}] no cells at {p} (claim map not run?)")
            st = p.stat()
            src = {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
            up = U.units_path(a, self.out)
            meta_p = up.with_suffix(".src.json")
            meta = read_json(meta_p)
            if meta and meta.get("src") == src and up.exists():
                us = list(U.iter_jsonl(up))
                h = meta["unit_hash"]
            else:
                us, _c = U.build_units(a, p)
                h = U.unit_set_hash(us)
                U.write_jsonl(up, us)  # a derived cache (also under --plan)
                write_json(meta_p, {"src": src, "unit_hash": h})
            hashes[a] = h
            allu.extend(us)
        return allu, hashes

    # ------------------------------------------------------------ §2 resample
    def pool_keys(self, units: list[dict[str, Any]]) -> dict[str, set[R.Key]]:
        out: dict[str, set[R.Key]] = defaultdict(set)
        for u in units:
            if u["arm"] in self.deterministic:
                continue
            p = A.pool_of(u["arm"], self.reg)
            if p is not None:
                out[p].add((u["id"], int(u["layer"]), int(u["pos"])))
        return dict(out)

    def resample_plan(self, units: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        res = {}
        for pool, want in self.pool_keys(units).items():
            st = R.load_state(pool, self.out)
            _complete, counts = R.merge_runs(R.load_runs(pool, self.out))
            row = self.reg["pools"][pool]
            spr = int(row.get("samples_per_run") or R.N_RESAMPLE)
            launches, failed = R.plan_launches(pool, want, counts, st, spr)
            req = launches[0].keys if launches else []
            pr = R.PoolRequest(pool, set(req), 0)
            # every listed position at every listed layer, samples_per_run per run
            n_gen_samples = pr.n_generated * len(launches) * spr
            pending = [r for r, v in st.runs.items() if v.get("status") in ("launched", "partial")]
            precheck = (R.pool_dir(pool, self.out) / "precheck.json").exists()
            res[pool] = {
                "n_keys": len(want),
                "n_complete": sum(1 for k in want if counts.get(k, 0) >= R.N_RESAMPLE),
                "n_request": len(req),
                "n_exhausted": len(failed),
                "n_launches": len(launches),
                "runs": [x.run for x in launches],
                "n_generated_cells": pr.n_generated if req else 0,
                "n_generated_samples": n_gen_samples if req else 0,
                "needs_precheck": bool(req) and not precheck,
                "pending_runs": pending,
                "usd": (n_gen_samples if req else 0)
                * float(row.get("usd_per_generated_sample") or 0.0),
            }
        return res

    def run_resample(self, units: list[dict[str, Any]]) -> None:
        plan = self.resample_plan(units)
        todo = {p: v for p, v in plan.items() if v["n_request"] or v["pending_runs"]}
        if not todo:
            return
        snap = check_snapshot(self.reg, "gw-mlnla", self.allow_snapshot_drift, self.log)
        want_all = self.pool_keys(units)
        deadline = time.time() + self.resample_timeout_s
        for pool in todo:
            # runs of an earlier (timed-out) gate whose Modal app is still running: poll them to
            # the end instead of launching a second attempt for their keys
            self.wait_in_flight(pool, deadline)
            self.resume_runs(pool, want_all[pool])
            pc = read_json(R.pool_dir(pool, self.out) / "precheck.json")
            if pc is not None:
                if not pc.get("ok", True):
                    self.failed_pools.add(pool)  # never launch full runs for this pool
                continue
            if not self.precheck(pool, sorted(want_all[pool])[:5], snap, deadline):
                self.failed_pools.add(pool)
        for _attempt in range(R.MAX_ATTEMPTS):
            failed = self.precheck_failed_pools()
            jobs = []
            for pool in todo:
                if pool in failed:
                    continue
                st = R.load_state(pool, self.out)
                _c, counts = R.merge_runs(R.load_runs(pool, self.out))
                spr = int(self.reg["pools"][pool].get("samples_per_run") or R.N_RESAMPLE)
                launches, _failed = R.plan_launches(pool, want_all[pool], counts, st, spr)
                for ln in launches:
                    jobs.append(self.start_launch(ln, snap, st))
                if launches:
                    st.record_launch(launches[0].keys)
                    R.save_state(st, self.out)
            if not jobs:
                break
            rcs = self.wait(jobs, deadline)
            for ln, _proc, _log in jobs:
                self.collect_launch(ln, rcs.get(ln.run))
        for pool in todo:
            R.rebuild_resample(pool, self.out)
        # a key still short with attempts left (its runs failed / returned nothing, so they did
        # not count): partial, never fill it as resample_failed
        failed = self.precheck_failed_pools()
        short = {}
        for pool in todo:
            if pool in failed:
                continue
            st = R.load_state(pool, self.out)
            _c, counts = R.merge_runs(R.load_runs(pool, self.out))
            spr = int(self.reg["pools"][pool].get("samples_per_run") or R.N_RESAMPLE)
            launches, _failed = R.plan_launches(pool, want_all[pool], counts, st, spr)
            if launches:
                short[pool] = len(launches[0].keys)
        if short:
            raise GateExit(
                EXIT_PARTIAL,
                f"resample incomplete with attempts left ({short}): failed / empty runs do not "
                "count as attempts; re-run resumes",
            )

    def run_app_ids(self, run: str) -> set[str]:
        import re

        lg = self.out.parent.parent / "logs" / f"gate_{run}.log"
        if not lg.exists():
            return set()
        return set(re.findall(r"\bap-[A-Za-z0-9]+", lg.read_text(errors="replace")))

    def in_flight_runs(self, pool: str) -> list[str]:
        """Runs in state launched / partial whose Modal app is still running (an app tied to
        the run by the app id in its launch log or by its description), or, when no app can be
        tied to it, whose volume dir changed in the last ``INFLIGHT_FRESH_S``."""
        st = R.load_state(pool, self.out)
        cand = [r for r, v in st.runs.items() if v.get("status") in ("launched", "partial")]
        if not cand:
            return []
        apps = self.modal.apps() or []
        mt: dict[str, float] | None = None
        now = time.time()
        out = []
        for r in cand:
            ids = self.run_app_ids(r)
            tied = [
                a
                for a in apps
                if (a["id"] and a["id"] in ids)
                or (a["description"] and (r in a["description"] or r[4:] in a["description"]))
            ]
            if tied:
                if any(R.app_running(a["state"]) for a in tied):
                    out.append(r)
                continue
            if mt is None:
                mt = self.modal.volume_mtimes(self.reg["env"]["WSBENCH_GEN_VOLUME"], "/")
            t = mt.get(r)
            if t is not None and now - t < INFLIGHT_FRESH_S:
                out.append(r)
        return out

    def wait_in_flight(self, pool: str, deadline: float) -> None:
        last = 0.0
        while True:
            fl = self.in_flight_runs(pool)
            if not fl:
                return
            if time.time() > deadline:
                raise GateExit(
                    EXIT_PARTIAL,
                    f"resample timeout: {len(fl)} earlier runs of {pool} still running "
                    f"({', '.join(fl)}); re-run resumes",
                )
            if time.time() - last >= HEARTBEAT_S:
                self.log(f"{pool}: waiting for {len(fl)} in-flight runs ({', '.join(fl[:3])})")
                last = time.time()
            time.sleep(POLL_S)

    def launch_dir(self, ln: R.Launch) -> Path:
        return R.pool_dir(ln.pool, self.out) / "launch" / ln.run

    def seed_launch(self, ln: R.Launch, subdir: str) -> list[int]:
        """Write the trimmed manifest for the launch's keys and seed ``subdir`` on the acts
        volume (volume put / cp: not launches)."""
        pool = R.pool_from_registry(ln.pool, self.reg)
        pr = R.PoolRequest(ln.pool, set(ln.keys), 0)
        man_src = read_json(Path(self.reg["acts_manifests"][pool.acts]))
        if man_src is None:
            raise GateExit(EXIT_FATAL, f"source manifest for {pool.acts} missing")
        pool_sub = R.Pool(pool.name, pool.arms, pool.generator, pool.acts, pool.lens)
        man, copies = R.trimmed_manifest(man_src, pool_sub, pr.positions())
        copies = [(a, str(Path("/", subdir, Path(b).name))) for a, b in copies]
        errs = R.check_manifest(man, pr.layers)
        if errs:
            raise GateExit(EXIT_FATAL, f"{ln.pool}: manifest errors {errs[:3]}")
        d = self.launch_dir(ln)
        write_json(d / "manifest.json", man, indent=None)
        write_json(d / "request.json", {**pr.counts(), "cells": sorted(pr.cells)}, indent=0)
        vol = self.reg["env"]["acts_volume"]
        if subdir not in self.seeded:  # seeded runs of one attempt share the subdir
            self.modal.volume(f"put --force {vol} {d / 'manifest.json'} /{subdir}/manifest.json")
            for a, b in copies:
                self.modal.volume(f"cp {vol} {a} {b}")
            self.seeded.add(subdir)
        return pr.layers

    def start_launch(self, ln: R.Launch, snap: Path, st: R.PoolState):
        subdir = f"acts-sibgate-{ln.pool}-{R.keys_sha8(ln.keys)}"
        layers = self.seed_launch(ln, subdir)
        cmd = R.generator_cmd(self.reg["pools"][ln.pool], ln, subdir, layers, self.modal.modal)
        log = self.out.parent.parent / "logs" / f"gate_{ln.run}.log"
        self.log(f"launch {ln.run} ({len(ln.keys)} keys, attempt {ln.attempt}): {cmd}")
        proc = self.modal.start(cmd, snap, log)
        st.runs[ln.run] = {
            "status": "launched",
            "attempt": ln.attempt,
            "n_keys": len(ln.keys),
            "keys_sha8": R.keys_sha8(ln.keys),
            "seed": ln.seed,
            "subdir": subdir,
            "ts": time.time(),
        }
        R.save_state(st, self.out)
        write_json(self.launch_dir(ln) / "keys.json", [list(k) for k in ln.keys], indent=None)
        return ln, proc, log

    def wait(self, jobs, deadline: float) -> dict[str, int | None]:
        """Poll the jobs to the end; returns run -> exit code."""
        last = time.time()
        while True:
            alive = [(ln, p, lg) for ln, p, lg in jobs if p.poll() is None]
            for ln, p, lg in jobs:
                rc = p.poll()
                if rc not in (None, 0):
                    tail = Path(lg).read_text()[-4000:] if Path(lg).exists() else ""
                    why = R.fatal_reason(tail)
                    if why:
                        raise GateExit(EXIT_FATAL, f"{ln.run}: Modal {why}: {tail[-600:]}")
            if not alive:
                return {ln.run: p.poll() for ln, p, _lg in jobs}
            if time.time() > deadline:
                raise GateExit(
                    EXIT_PARTIAL,
                    f"resample timeout: {len(alive)} runs still running "
                    f"({', '.join(x.run for x, _p, _l in alive)}); re-run resumes",
                )
            if time.time() - last >= HEARTBEAT_S:
                self.log(f"heartbeat: {len(alive)} generator runs running")
                last = time.time()
            time.sleep(POLL_S)

    def collect_launch(self, ln: R.Launch, rc: int | None = None) -> int:
        """Fetch + collect one run. A run with a non-zero exit code, a failed volume get or zero
        collected rows does not count as an attempt for its keys (``PoolState.void``)."""
        vol = self.reg["env"]["WSBENCH_GEN_VOLUME"]
        g = R.pool_dir(ln.pool, self.out) / "gen"
        got = bool(self.modal.get(vol, f"/{ln.run}", g))
        rows = R.collect_run_rows(g / ln.run, ln.keys, ln.seed) if got else []
        st = R.load_state(ln.pool, self.out)
        if got:
            U.write_jsonl(R.runs_dir(ln.pool, self.out) / f"{ln.run}.jsonl", rows)
        # a run fetched before it finished (exit 4, then a re-run) stays "partial" and is
        # re-fetched next time; runs/<run>.jsonl is rewritten, never appended (no double count)
        done = {(r["id"], int(r["layer"]), int(r["pos"])) for r in rows} >= set(ln.keys)
        bad = rc not in (None, 0) or not got or not rows
        run = st.runs.setdefault(ln.run, {})
        run["status"] = "failed" if bad else ("collected" if done else "partial")
        run["n_rows"] = len(rows)
        if bad:
            why = (
                f"rc {rc}"
                if rc not in (None, 0)
                else ("volume get failed" if not got else "0 rows")
            )
            run["why"] = why
            if ln.attempt > 0 and run.get("attempt") is not None:
                st.void(ln.keys, ln.attempt)
                self.log(f"{ln.run}: {why}: attempt {ln.attempt} not counted")
        R.save_state(st, self.out)
        self.log(f"collected {ln.run}: {len(rows)} rows")
        return len(rows)

    def resume_runs(self, pool: str, want: set[R.Key]) -> None:
        """Volume-listed runs of this pool that were launched but never collected."""
        st = R.load_state(pool, self.out)
        have = set(R.collected_runs(pool, self.out))
        names = self.modal.ls(self.reg["env"]["WSBENCH_GEN_VOLUME"], "/")
        for n in names:
            pr = R.parse_run_name(n, self.reg["pools"])
            if pr is None or pr["pool"] != pool:
                continue
            if n in have and (st.runs.get(n) or {}).get("status") == "collected":
                continue
            kp = R.pool_dir(pool, self.out) / "launch" / n / "keys.json"
            keys = [tuple(k) for k in read_json(kp, [])] or sorted(want)
            seed = (st.runs.get(n) or {}).get("seed")
            if seed is None and pr["r"] is not None:
                seed = R.seed_of(n)
            self.collect_launch(R.Launch(pool, n, pr["attempt"], keys, seed))

    def precheck(self, pool: str, keys: list[R.Key], snap: Path, deadline: float) -> bool:
        """§2: the 5-cell key + shape check, once per new pool, before its first full launch."""
        sha = R.keys_sha8(keys)
        name = f"gen-sibgate-{pool}-precheck-{sha}"
        row = self.reg["pools"][pool]
        spr = int(row.get("samples_per_run") or R.N_RESAMPLE)
        lns = (
            [R.Launch(pool, name, 0, keys)]
            if spr >= R.N_RESAMPLE
            else [
                R.Launch(pool, f"{name}-r{j}", 0, keys, seed=R.seed_of(f"{name}-r{j}"))
                for j in range(2)
            ]
        )
        jobs = []
        for ln in lns:
            subdir = f"acts-sibgate-{pool}-precheck-{sha}"
            layers = self.seed_launch(ln, subdir)
            cmd = R.generator_cmd(row, ln, subdir, layers, self.modal.modal)
            log = self.out.parent.parent / "logs" / f"gate_{ln.run}.log"
            self.log(f"precheck {pool}: {cmd}")
            jobs.append((ln, self.modal.start(cmd, snap, log), log))
        self.wait(jobs, deadline)
        g = R.pool_dir(pool, self.out) / "precheck_gen"
        new: dict[R.Key, list[str]] = defaultdict(list)
        for ln in lns:
            self.modal.get(self.reg["env"]["WSBENCH_GEN_VOLUME"], f"/{ln.run}", g)
            for r in R.collect_run_rows(g / ln.run, keys):
                new[(r["id"], int(r["layer"]), int(r["pos"]))].extend(r["samples"])
        orig = {
            (r["id"], int(r["layer"]), int(r["pos"])): r["samples"]
            for r in U.iter_jsonl(Path(row["orig_readouts"]))
        }
        ks = [k for k in keys if k in orig and k in new]
        shape = R.block_shape_check(
            [s for k in ks for s in orig[k]], [s for k in ks for s in new[k]]
        )
        ok = set(new) >= set(keys) and shape["ok"]
        write_json(
            R.pool_dir(pool, self.out) / "precheck.json",
            {
                "ok": ok,
                "keys": [list(k) for k in keys],
                "n_collected": len(new),
                "block_shape": shape,
                "runs": [x.run for x in lns],
                "ts": time.time(),
            },
        )
        self.log(f"precheck {pool}: {'OK' if ok else 'FAILED -> resample_failed'}")
        return ok

    def resampled(self) -> dict[str, dict[R.Key, list[str]]]:
        return {p: R.load_resample(p, self.out) for p in self.reg["pools"]}

    def precheck_failed_pools(self) -> set[str]:
        out = set(self.failed_pools)
        for p in self.reg["pools"]:
            pc = read_json(R.pool_dir(p, self.out) / "precheck.json")
            if pc is not None and not pc.get("ok", True):
                out.add(p)
        return out

    # ------------------------------------------------------------ §4 fill (rung 1)
    def fill_plan(self, units, resampled) -> dict[str, Any]:
        from hallucination_bench.cache import Cache, fingerprint

        from . import fill_judge as F

        n_calls = n_pending_resample = n_failed_kept = 0
        fp_cache = Cache(self.out / "fill" / "cache.jsonl", readonly=True)
        its = self.items()
        failed = self.precheck_failed_pools()
        exhausted: dict[str, R.PoolState] = {}
        for u in units:
            if u["arm"] in self.deterministic:
                continue
            pool = A.pool_of(u["arm"], self.reg)
            blocks = (resampled.get(pool) or {}).get((u["id"], int(u["layer"]), int(u["pos"])))
            if pool in failed:
                continue
            if not blocks:
                k = (u["id"], int(u["layer"]), int(u["pos"]))
                if exhausted.get(pool) is None:
                    exhausted[pool] = R.load_state(pool, self.out)
                if not exhausted[pool].spent(k):
                    n_pending_resample += 1  # resample_failed once its attempts are spent
                continue
            user = F.render_user(u, F.conversation_for(its[u["id"]], u["pos"]), blocks)
            fp = fingerprint(F.PROMPT_VERSION, F.MODEL, F.EFFORT, F.SYSTEM, user)
            key = U.fill_cache_key(u)
            if fp_cache.get(key, fp) is not None:
                continue
            if fp_cache.get(key, fp, include_failed=True) is not None and not self.retry_failed:
                n_failed_kept += 1
                continue
            n_calls += 1
        fp_cache.close()
        return {
            "n_calls": n_calls,
            "n_after_resample": n_pending_resample,
            "n_failed_kept": n_failed_kept,
        }

    def run_fill(self, units, resampled) -> dict[str, dict[str, Any]]:
        from hallucination_bench.cache import Cache

        from . import fill_judge as F

        failed = frozenset(self.precheck_failed_pools())

        async def go():
            with Cache(self.out / "fill" / "cache.jsonl") as cache:
                return await F.run_fill(
                    units,
                    resampled,
                    self.items(),
                    self.llm,
                    cache,
                    failed_pools=failed,
                    deterministic=frozenset(self.deterministic),
                    pool_of=lambda a: A.pool_of(a, self.reg),
                )

        rows, spend = asyncio.run(go())
        self.spend_usd += spend.usd
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by[r["arm"]].append(r)
        for arm, rs in by.items():
            U.write_jsonl(F.rung1_path(arm, self.out), rs)
        return {r["unit"]: r for r in rows}

    def load_rung1(self) -> dict[str, dict[str, Any]]:
        from . import fill_judge as F

        out: dict[str, dict[str, Any]] = {}
        for a in self.gated:
            p = F.rung1_path(a, self.out)
            if p.exists():
                out.update({r["unit"]: r for r in U.iter_jsonl(p)})
        return out

    # ------------------------------------------------------------ §4 classes
    def class_pairs(self, units, rung1) -> dict[tuple[str, str], dict[str, Any]]:
        from . import classes as K
        from . import fill_judge as F

        return K.class_requests([u for u in units if F.needs_rung2(rung1.get(u["unit"]) or {})])

    def run_classes(self, reqs) -> dict[str, dict[str, Any]]:
        """Frozen pairs from ``class_map.json``; new pairs -> ``sibclass-v1``; new raw names ->
        one ``dedupe-incr-v1`` call. Writes class_map.json (when it changes) and classes.json."""
        from hallucination_bench.cache import Cache

        from . import classes as K

        cm = K.load_class_map(self.data)
        if cm is None:  # a fresh gate (no plan-0009 import): an empty frozen map
            self.log(f"{K.class_map_path(self.data)} missing: starting an empty class map")
            cm = {"version": K.CLASS_MAP_VERSION, "names": {}, "pairs": {}}
        frozen = K.raw_of_map(cm)
        names: dict[str, str] = dict(cm["names"])
        new = {
            k: r
            for k, r in reqs.items()
            if k not in frozen or (frozen[k] is None and self.retry_failed)
        }
        raw = dict(frozen)
        changed = False
        if new:
            self.log(f"classes: {len(new)} new (said, item) pairs -> sibclass-v1")

            async def go():
                with Cache(self.out / "classes" / "cache.jsonl") as cache:
                    got, spend = await K.run_classes(new, self.items(), self.llm, cache)
                    fresh = K.new_raw_names(got, names)
                    dd, frz = await K.run_dedupe_incr(
                        fresh, sorted(set(names.values())), self.llm, cache, spend
                    )
                    return got, dd, frz, spend

            got, dd, frz, spend = asyncio.run(go())
            self.spend_usd += spend.usd
            raw.update(got)
            for k, r in got.items():
                cm["pairs"][f"{k[0]}|{k[1]}"] = r
            if frz:
                names.update(dd)
                cm["names"] = names
            else:
                self.log(f"dedupe-incr failed: {len(dd)} names unfrozen (re-asked next run)")
            changed = True
        cur = {k: raw.get(k) for k in reqs}
        all_names = {r["class"] for r in cur.values() if r} - {K.NO_CLASS, ""}
        dd_cur = {n: names.get(n, n) for n in sorted(all_names)}
        classes = K.assemble(cur, dd_cur)
        if changed:
            K.save_class_map(cm, self.data)
        self.write_if_changed(
            self.data / "classes.json",
            {"version": K.CLASS_VERSION, "dedupe": dd_cur, "classes": classes},
        )
        return classes

    def write_if_changed(self, p: Path, obj: Any, indent: int = 1) -> bool:
        txt = json.dumps(obj, indent=indent, ensure_ascii=False) + "\n"
        if p.exists() and p.read_text() == txt:
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(txt)
        tmp.replace(p)
        return True

    # ------------------------------------------------------------ §4 concepts (frozen fillers)
    def kept_classes(self, classes) -> dict[str, int]:
        from . import classes as K

        n = Counter(v["class"] for v in classes.values() if v["class"] != K.NO_CLASS)
        return {c: k for c, k in n.items() if k >= MIN_CLASS_PAIRS}

    def concepts_plan(self, classes) -> dict[str, Any]:
        from . import classes as K

        cur = read_json(self.data / "concepts.json", {"classes": {}})["classes"]
        kept = self.kept_classes(classes)
        members = K.class_members({k: v for k, v in classes.items() if v["class"] in kept})
        crossing = sorted(c for c in kept if c not in cur)
        appended = {}
        for c, m in members.items():
            if c in cur:
                have = {U.norm(x) for x in cur[c]["concepts"]}
                add = sorted(
                    {x for x in m["said"] | m["gold"] if U.norm(x) not in have}, key=str.casefold
                )
                if add:
                    appended[c] = add
        return {"crossing": crossing, "appended": appended, "members": members, "cur": cur}

    def run_concepts(self, classes) -> dict[str, dict[str, Any]]:
        from hallucination_bench.cache import Cache

        from . import classes as K

        pl = self.concepts_plan(classes)
        cur = dict(pl["cur"])
        for c, add in pl["appended"].items():
            row = dict(cur[c])
            row["concepts"] = [*row["concepts"], *add]
            m = pl["members"][c]
            row["said"], row["gold"] = sorted(m["said"]), sorted(m["gold"])
            row.setdefault("appended", []).extend(add)
            cur[c] = row
            self.log(f"concepts: {c}: appended {len(add)} members (fillers frozen)")
        if pl["crossing"]:
            self.log(
                f"concepts: {len(pl['crossing'])} classes cross the {MIN_CLASS_PAIRS}-pair cap"
            )

            async def go():
                with Cache(self.out / "classes" / "cache.jsonl") as cache:
                    return await K.run_concepts(
                        {c: pl["members"][c] for c in pl["crossing"]}, self.llm, cache
                    )

            got = asyncio.run(go())
            cur.update(got)
        comp = K.compatible_of(classes)
        obj = {
            "version": K.CONCEPTS_VERSION,
            "classes": dict(sorted(cur.items())),
            "compatible": {g: sorted(v) for g, v in comp.items()},
        }
        self.write_if_changed(self.data / "concepts.json", obj)
        return obj["classes"]

    # ------------------------------------------------------------ §4 text bank
    def texts_plan(self, concepts) -> dict[str, list[tuple[str | None, str]]]:
        from . import texts as T

        idx = T.load_cells_index(self.data)
        return {
            c: m for c, row in concepts.items() if (m := T.missing_cells(c, row["concepts"], idx))
        }

    def run_texts(self, units, classes, concepts) -> list[str]:
        from hallucination_bench.cache import Cache

        from . import ladder as L
        from . import texts as T
        from .stages import _surface

        todo = self.texts_plan(concepts)
        if not todo:
            return []
        self.log(f"texts: {sum(len(v) for v in todo.values())} new cells in {len(todo)} classes")
        existing = {c: T.load_bank_class(self.data, L.slug(c)) for c in todo}
        surface = _surface(units, classes)
        index = T.item_ngram_index(self.items().values())

        async def go():
            with Cache(self.out / "texts" / "cache.jsonl") as cache:
                return await T.run_text_cells(
                    todo, concepts, surface, index, existing, self.llm, cache
                )

        rows, cells, causes = asyncio.run(go())
        idx = T.load_cells_index(self.data)
        for c in todo:
            # per class: the bank file, then the index right after it (a crash between classes
            # leaves every written class consistent with its attempted cells)
            T.write_bank_class(self.data, L.slug(c), [*existing[c], *rows.get(c, [])])
            idx.setdefault(c, []).extend(cells.get(c, []))
            T.save_cells_index(self.data, idx)
        self.log(f"texts: {sum(len(v) for v in rows.values())} kept; drops {dict(causes)}")
        return sorted(todo)

    # ------------------------------------------------------------ §5 capture (append-only)
    def capture_plan(self, concepts) -> dict[str, int]:
        from . import ladder as L
        from . import texts as T

        out = {}
        for c in concepts:
            rows = T.load_bank_class(self.data, L.slug(c))
            n = sum(1 for t in rows if not t.get("capture")) if c not in self.remerge else len(rows)
            if n:
                out[c] = n
        return out

    def get_tokenizer(self):
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore[import-not-found]
            except ImportError:
                raise GateExit(
                    EXIT_FATAL,
                    "capture needs transformers: uv run --with transformers --with jinja2 "
                    "hbench gate",
                ) from None
            self.tokenizer = AutoTokenizer.from_pretrained(self.reg["env"]["tokenizer"])
        return self.tokenizer

    def encode(self, t: Mapping[str, Any]):
        from . import texts as T

        tok = self.get_tokenizer()
        msgs = [
            {"role": "system", "content": t["system"]},
            {"role": "user", "content": t["user"]},
            {"role": "assistant", "content": t["assistant"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        reply_at = text.rfind(t["assistant"])
        text = text[: reply_at + len(t["assistant"])]
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        toks = [text[s:e] for s, e in enc["offset_mapping"]]
        starts = [s for s, _e in enc["offset_mapping"]]
        lo = reply_at + int(t["mention_char"] or 0)
        raw = [T.token_class(toks[i]) for i in T.candidates(toks, starts, reply_at, lo)]
        return toks, starts, reply_at, lo, raw

    def run_capture(self, concepts) -> list[str]:
        from . import ladder as L
        from . import texts as T
        from .stages import probe_label

        todo = self.capture_plan(concepts)
        if not todo:
            return []
        merges = read_json(self.probes / "merge.json", {})
        prompts, owners = [], []
        for c in sorted(todo):
            rows = T.load_bank_class(self.data, L.slug(c))
            sel = [i for i, t in enumerate(rows) if c in self.remerge or not t.get("capture")]
            enc = {i: self.encode(rows[i]) for i in sel}
            if c not in merges or c in self.remerge:
                imp = [
                    (self.encode(t) if i not in enc else enc[i])[4]
                    for i, t in enumerate(rows)
                    if t["concept"] and t["kind"] == "implied"
                ]
                m = T.class_merge(imp)
                merges[c] = [m[k] for k in range(T.N_CLASSES)]
                self.log(f"capture: merge {'re-' if c in self.remerge else ''}frozen for {c}")
            merge = {k: int(v) for k, v in enumerate(merges[c])}
            for i in sel:
                t = rows[i]
                toks, starts, reply_at, lo, _raw = enc[i]
                pos = T.read_positions(toks, starts, reply_at, lo, merge)
                _e, key = T.class_key([toks[p] for p in pos], merge)
                prompts.append(
                    {
                        "label": probe_label(t["id"]),
                        "family": "sibgate_probe",
                        "system": t["system"],
                        "user": t["user"],
                        "prefill": t["assistant"],
                        "look_for": "",
                        "targets": [],
                        "eval_positions": pos,
                        "meta": {
                            "class": c,
                            "concept": t["concept"],
                            "kind": t["kind"],
                            "key": key,
                        },
                    }
                )
                owners.append((c, i, key))
        sha = sha256_json([p["label"] for p in prompts])[:8]
        # split into parallel sub-shards (plan-0009 used 32 by hand; one container takes ~3.5 h
        # for 44k prompts). k = 1 keeps the plain name. Every launch passes --resume, so a
        # re-run (or a preempted container's restart) keeps the per-prompt files already written.
        k = self.capture_shards or max(1, math.ceil(len(prompts) / CAPTURE_PER_SHARD))
        parts = [prompts[j::k] for j in range(k)]
        names = [
            f"acts-sibgate-probe-{sha}" if k == 1 else f"acts-sibgate-probe-{sha}-k{j:02d}"
            for j in range(k)
        ]
        shard_of = {p["label"]: names[j] for j, part in enumerate(parts) for p in part}
        snap = check_snapshot(self.reg, "gw-wrong-bridge", self.allow_snapshot_drift, self.log)
        from concurrent.futures import ThreadPoolExecutor

        from . import ladder as Ld

        def one(j: int) -> tuple[str, int, str]:
            cp = (
                self.probes / "shards" / (f"cp_{sha}.json" if k == 1 else f"cp_{sha}_k{j:02d}.json")
            )
            write_json(cp, parts[j], indent=None)
            cmd = (
                f"PYTHONPATH=src {self.modal.modal} run "
                "scripts/olens_suite/workspace_bench/wsbench_capture_modal.py "
                f"--prompts-json {cp.resolve()} --acts-subdir {names[j]} --resume "
                f"--layers {','.join(str(x) for x in Ld.PROBE_LAYERS)}"
            )
            rc, txt = self.modal.run(cmd, snap)
            return names[j], rc, txt

        self.log(f"capture: {len(prompts)} prompts -> {k} shard(s) {names[0]}..")
        with ThreadPoolExecutor(max_workers=k) as ex:
            res = list(ex.map(one, range(k)))
        bad = [(n, rc, txt) for n, rc, txt in res if rc != 0]
        if bad:
            n, rc, txt = bad[0]
            raise GateExit(
                EXIT_PARTIAL,
                f"capture: {len(bad)}/{k} shard(s) failed; first {n} (rc {rc}): {txt[-600:]}",
            )
        write_json(self.probes / "merge.json", merges, indent=0)
        by: dict[str, dict[int, list[int]]] = defaultdict(dict)
        for c, i, key in owners:
            by[c][i] = key
        for c, keys in by.items():
            rows = T.load_bank_class(self.data, L.slug(c))
            for i, key in keys.items():
                lab = probe_label(rows[i]["id"])
                rows[i]["capture"] = {"label": lab, "shard": shard_of[lab], "key": key}
            if c in self.remerge:  # recaptured: every text of the class sits in the new shards
                for t in rows:
                    t["capture"]["shard"] = shard_of.get(
                        t["capture"]["label"], t["capture"]["shard"]
                    )
            T.write_bank_class(self.data, L.slug(c), rows)
        return sorted(by)

    # ------------------------------------------------------------ §5 fit (per-class hash)
    def fit_state(self, concepts) -> dict[str, dict[str, Any]]:
        """Per class: the current input hash, the stored one, and the slice."""
        from . import ladder as L
        from . import texts as T

        merges = read_json(self.probes / "merge.json", {})
        out = {}
        for c, row in sorted(concepts.items()):
            slug = L.slug(c)
            rows = T.load_bank_class(self.data, slug)
            merge = merges.get(c, list(range(T.N_CLASSES)))
            sl = fit_slice(c, row, rows, merge)
            shards = class_shards(rows)
            h = fit_input_hash(sl, rows, shards)
            hp = self.probes / "fit" / slug / "inputs.sha256"
            old = hp.read_text().strip() if hp.exists() else None
            out[c] = {"slug": slug, "hash": h, "old": old, "slice": sl, "shards": shards}
        return out

    def run_fit(self, concepts) -> list[str]:
        fs = self.fit_state(concepts)
        todo = [c for c, s in fs.items() if s["hash"] != s["old"]]
        if not todo:
            return []
        from . import ladder as L

        inp = {
            "run": self.reg["env"]["probe_run"],
            "probe_acts": ",".join(sorted({x for c in todo for x in fs[c]["shards"]})),
            "layers": list(L.PROBE_LAYERS),
            # plan 0010 §5: each class reads its own shard list (append-only capture)
            "classes": {
                fs[c]["slug"]: {**fs[c]["slice"], "probe_acts": ",".join(fs[c]["shards"])}
                for c in todo
            },
        }
        p = self.probes / f"probe_inputs_{sha256_json(sorted(todo))[:8]}.json"
        write_json(p, inp, indent=None)
        self.log(f"fit: {len(todo)} classes (inputs changed)")
        rc, txt = self.modal.run(
            f"{self.modal.modal} run {SCRIPTS / 'probes_modal.py'} --phase fit --inputs {p} "
            f"--out-dir {self.probes}",
            REPO_ROOT,
        )
        if rc != 0:
            raise GateExit(EXIT_PARTIAL, f"fit failed (rc {rc}): {txt[-600:]}")
        for c in todo:
            hp = self.probes / "fit" / fs[c]["slug"] / "inputs.sha256"
            hp.parent.mkdir(parents=True, exist_ok=True)
            hp.write_text(fs[c]["hash"] + "\n")
        return todo

    # ------------------------------------------------------------ §5 distractors (first freeze)
    def load_distractors(self) -> dict[str, dict[str, Any]]:
        d = self.data / "distractors"
        out: dict[str, dict[str, Any]] = {}
        if d.exists():
            for f in sorted(d.glob("*.json")):
                out.update(json.loads(f.read_text()))
        return out

    def freeze_plan(self, units, rung1, classes, concepts) -> list[dict[str, Any]]:
        from . import fill_judge as F

        dis = self.load_distractors()
        out = []
        for u in units:
            if u["unit"] in dis or not F.needs_rung2(rung1.get(u["unit"]) or {}):
                continue
            v = classes.get(f"{u['said_norm']}|{u['id']}")
            if v and v["class"] != "no_class" and v["class"] in concepts:
                out.append(u)
        return out

    def run_freeze(self, units, rung1, classes, concepts, cps, generation: int) -> list[str]:
        from . import ladder as L
        from .stages import _distractors

        todo = self.freeze_plan(units, rung1, classes, concepts)
        if not todo:
            return []
        want = {u["unit"] for u in todo}
        r1 = {k: v for k, v in rung1.items() if k in want}
        got = _distractors(units, r1, classes, concepts, cps)
        by: dict[str, dict[str, Any]] = defaultdict(dict)
        for unit, rec in got.items():
            u = next(x for x in todo if x["unit"] == unit)
            cls = classes[f"{u['said_norm']}|{u['id']}"]["class"]
            by[L.slug(cls)][unit] = {**rec, "sha": distractor_sha(rec), "generation": generation}
        for slug, recs in by.items():
            p = self.data / "distractors" / f"{slug}.json"
            cur = read_json(p, {})
            for k, v in recs.items():
                cur.setdefault(k, v)  # an existing unit's set is never replaced
            write_json(p, dict(sorted(cur.items())), indent=0)
        self.log(f"distractors: first freeze of {len(got)} units in {len(by)} classes")
        return sorted(by)

    # ------------------------------------------------------------ §10.6 score (per class meta)
    def score_state(self, concepts) -> dict[str, dict[str, Any]]:
        from . import ladder as L

        dis = self.load_distractors_by_class()
        es = eval_sites_hash(self.items())
        out = {}
        for c in sorted(concepts):
            slug = L.slug(c)
            fd = self.probes / "fit" / slug
            ih = (
                (fd / "inputs.sha256").read_text().strip()
                if (fd / "inputs.sha256").exists()
                else None
            )
            fits = [read_json(f) for f in sorted(fd.glob("fit_L*.json"))] if fd.exists() else []
            scorable = bool(fits) and not all(f.get("error") for f in fits)
            meta = read_json(self.probes / "score" / slug / "meta.json")
            want = {
                "class": c,
                "input_hash": ih,
                "eval_sites": es,
                "units": {k: v["sha"] for k, v in sorted(dis.get(slug, {}).items())},
                "scored": scorable,
            }
            stale = meta is None or any(
                meta.get(k) != want[k] for k in ("input_hash", "eval_sites", "units")
            )
            out[c] = {"slug": slug, "want": want, "meta": meta, "stale": stale and ih is not None}
        return out

    def load_distractors_by_class(self) -> dict[str, dict[str, Any]]:
        d = self.data / "distractors"
        return (
            {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}
            if d.exists()
            else {}
        )

    def run_score(self, concepts, generation: int) -> list[str]:
        ss = self.score_state(concepts)
        todo = [c for c, s in ss.items() if s["stale"]]
        if not todo:
            self.rebuild_score_summary(ss)
            return []
        todo_s = [c for c in todo if ss[c]["want"]["scored"]]
        if todo_s:
            from . import ladder as L
            from . import texts as T

            its = self.items()
            dis = self.load_distractors_by_class()
            dp = self.probes / "score" / f"distractors_{generation}.json"
            write_json(
                dp, {k: v for c in todo_s for k, v in dis.get(ss[c]["slug"], {}).items()}, indent=0
            )
            inp = {
                "run": self.reg["env"]["probe_run"],
                "items": sorted(its),
                "site_keys": {
                    it: {
                        str(s["pos"]): T.token_class(s.get("token") or "") for s in its[it]["sites"]
                    }
                    for it in its
                },
                "eval_acts": eval_acts(),
                "layers": list(L.PROBE_LAYERS),
                "classes": sorted(ss[c]["slug"] for c in todo_s),
            }
            p = self.probes / f"score_inputs_{generation}.json"
            write_json(p, inp, indent=None)
            self.log(f"score: {len(todo_s)} classes (inputs / eval sites / unit shas changed)")
            rc, txt = self.modal.run(
                f"{self.modal.modal} run {SCRIPTS / 'probes_modal.py'} --phase score --inputs {p} "
                f"--out-dir {self.probes} --distractors {dp}",
                REPO_ROOT,
            )
            if rc != 0:
                raise GateExit(EXIT_PARTIAL, f"score failed (rc {rc}): {txt[-600:]}")
        for c in todo:
            write_json(
                self.probes / "score" / ss[c]["slug"] / "meta.json",
                {**ss[c]["want"], "generation": generation, "ts": time.time()},
            )
        self.rebuild_score_summary(self.score_state(concepts))
        return todo

    def rebuild_score_summary(self, ss) -> None:
        summ = {
            "run": self.reg["env"]["probe_run"],
            "rebuilt_from": "per-class meta.json (plan 0010 §10.6)",
            "classes": {
                s["slug"]: {
                    "input_hash": (s["meta"] or {}).get("input_hash"),
                    "n_units": len((s["meta"] or {}).get("units") or {}),
                    "scored": (s["meta"] or {}).get("scored"),
                    "generation": (s["meta"] or {}).get("generation"),
                }
                for s in ss.values()
            },
        }
        if not self.plan_only:
            write_json(self.probes / "score" / "summary.json", summ)

    # ------------------------------------------------------------ label
    def stale_units(self, units, classes, concepts) -> set[str]:
        from . import ladder as L

        dis = self.load_distractors()
        metas: dict[str, dict[str, Any] | None] = {}
        out = set()
        for u in units:
            rec = dis.get(u["unit"])
            if rec is None:
                continue
            v = classes.get(f"{u['said_norm']}|{u['id']}")
            if not v or v["class"] not in concepts:
                continue
            slug = L.slug(v["class"])
            if slug not in metas:
                metas[slug] = read_json(self.probes / "score" / slug / "meta.json")
            m = metas[slug]
            if m is None:
                continue
            if (m.get("units") or {}).get(u["unit"]) != rec.get("sha"):
                out.add(u["unit"])
        return out

    def run_label(self, units, rung1, classes, concepts, cps):
        from .stages import label_rows

        dis = self.load_distractors()
        stale = self.stale_units(units, classes, concepts)
        if stale:
            self.log(f"label: {len(stale)} units stale_score (distractor sha != score meta)")
        ready = bool(classes) and any(cp.z is not None for cp in cps.values())
        rows, nulls = label_rows(
            units,
            rung1,
            classes,
            dis,
            cps,
            site_tokens(self.items()),
            ready=ready,
            stale=stale,
        )
        return rows, nulls, ready

    # ------------------------------------------------------------ the generation stamp (§6)
    def stamp(self, unit_hashes, concepts, fit_hashes, rows=None) -> dict[str, Any]:
        """The generation stamp; ``rows`` (the label rows) enter as a hash of every
        (unit, rung1, final), so a changed outcome always bumps the generation."""
        from . import ladder as L
        from . import texts as T

        bank_files = sorted((T.bank_dir(self.data)).glob("*.jsonl.gz"))
        outcomes = [[r["unit"], r.get("rung1") or "", r.get("final") or ""] for r in (rows or [])]
        return {
            "gated_arms": list(self.gated),
            "class_map": sha256_file(self.data / "class_map.json"),
            "classes": sha256_file(self.data / "classes.json"),
            "concepts": sha256_file(self.data / "concepts.json"),
            "bank": sha256_json(
                [(f.name, sha256_file(f)) for f in bank_files]
                + [("cells.json", sha256_file(T.bank_dir(self.data) / "cells.json"))]
            ),
            "fit": sha256_json(fit_hashes),
            "merge": sha256_file(self.probes / "merge.json"),
            "distractors": sha256_json(
                [
                    (f.name, sha256_file(f))
                    for f in sorted((self.data / "distractors").glob("*.json"))
                ]
            ),
            "unit_hash": dict(unit_hashes),
            "outcomes": sha256_json(sorted(outcomes or [])),
            "rung1_skipped": list(self.skipped_rung1),
            "probe_layers": list(L.PROBE_LAYERS),
        }

    # ------------------------------------------------------------ cost (§6)
    def unit_costs(self) -> dict[str, Any]:
        return read_json(self.data / "unit_costs.json", {}) or {}

    # ------------------------------------------------------------ plan
    def plan(self) -> dict[str, Any]:
        """All the work, without side effects (no calls, no launches, no writes)."""
        from . import fill_judge as F

        was = self.plan_only
        self.plan_only = True
        try:
            self.check_registry()
            units, uh = self.load_units()
            prev = read_json(self.out / "generation.json", {}) or {}
            prev_uh = (prev.get("stamp") or {}).get("unit_hash") or {}
            res = self.resample_plan(units)
            resampled = self.resampled()
            fill = self.fill_plan(units, resampled)
            rung1 = self.load_rung1()
            pairs = self.class_pairs(units, rung1)
            from . import classes as K

            cm = K.load_class_map(self.data) or {"pairs": {}, "names": {}}
            frozen = K.raw_of_map(cm) if cm["pairs"] else {}
            new_pairs = [k for k in pairs if k not in frozen]
            retry_pairs = (
                [k for k in pairs if frozen.get(k, 1) is None] if self.retry_failed else []
            )
            classes = read_json(self.data / "classes.json", {"classes": {}})["classes"]
            cpl = self.concepts_plan(classes)
            concepts = read_json(self.data / "concepts.json", {"classes": {}})["classes"]
            tpl = self.texts_plan(concepts)
            cap = self.capture_plan(concepts)
            fs = self.fit_state(concepts)
            refit = sorted(c for c, s in fs.items() if s["hash"] != s["old"])
            freeze = self.freeze_plan(units, rung1, classes, concepts)
            ss = self.score_state(concepts)
            rescore = sorted(c for c, s in ss.items() if s["stale"])
            # units with no rung-1 row yet (new arms): rung 2 is estimated, not known
            n_unknown = sum(1 for u in units if u["unit"] not in rung1)
            flips = self.flip_sources(units, rung1, prev)
            est = self.estimate(units, rung1, res, fill, new_pairs, cpl, tpl, cap, refit, rescore)
            arms_rows = {
                a: {
                    "unit_hash": uh[a][:12],
                    "prev": (prev_uh.get(a) or "")[:12] or None,
                    "changed": prev_uh.get(a) != uh[a],
                    "n_units": sum(1 for u in units if u["arm"] == a),
                }
                for a in self.gated
            }
            work = {
                "gated_arms": list(self.gated),
                "added": list(self.added),
                "arms": arms_rows,
                "resample": res,
                "fill": fill,
                "classes": {
                    "n_pairs": len(pairs),
                    "new_pairs": len(new_pairs),
                    "retry_pairs": len(retry_pairs),
                },
                "concepts": {
                    "crossing": cpl["crossing"],
                    "appended": {c: len(v) for c, v in cpl["appended"].items()},
                },
                "texts": {c: len(v) for c, v in tpl.items()},
                "capture": cap,
                "remerge": list(self.remerge),
                "refit": refit,
                "first_freeze": len(freeze),
                "rescore": rescore,
                "n_units_without_rung1": n_unknown,
                "flip_sources": flips,
                "estimate": est,
                "needs_rung2_known": sum(
                    1 for u in units if F.needs_rung2(rung1.get(u["unit"]) or {})
                ),
            }
            work["zero_work"] = not (
                any(v["n_request"] or v["pending_runs"] for v in res.values())
                or fill["n_calls"]
                or fill["n_after_resample"]
                or new_pairs
                or retry_pairs
                or cpl["crossing"]
                or cpl["appended"]
                or tpl
                or cap
                or refit
                or freeze
                or rescore
                or n_unknown
            )
            self.work = work
            return work
        finally:
            self.plan_only = was

    def flip_sources(self, units, rung1, prev) -> dict[str, Any]:
        """Expected flip sources for previously gated arms: cross-index entries that new units add
        to an existing unit's (item, said); classes to refit; cap crossings."""
        from . import classes as K

        classes = read_json(self.data / "classes.json", {"classes": {}})["classes"]
        canon = K.canonical_map(classes)
        prev_arms = set((prev.get("stamp") or {}).get("gated_arms") or [])
        new_arms = [a for a in self.gated if a not in prev_arms] if prev_arms else []
        by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        for u in units:
            c = canon.get((u["said_norm"], u["id"]), u["said_norm"])
            by_key[(u["id"], c)].add(u["arm"])
        touched = 0
        for u in units:
            if u["arm"] in new_arms:
                continue
            c = canon.get((u["said_norm"], u["id"]), u["said_norm"])
            if by_key[(u["id"], c)] & set(new_arms):
                touched += 1
        return {
            "new_arms": new_arms,
            "cross_method_touched_units": touched,
            "note": "class refits / cap crossings from new pairs are known only after the "
            "classes stage"
            if new_arms
            else "",
        }

    def estimate(self, units, rung1, res, fill, new_pairs, cpl, tpl, cap, refit, rescore):
        """The plan-0009 unit costs x the counts; for units without rung 1 yet (a new arm),
        the downstream counts are estimated with the plan-0009 rates (marked ``est``)."""
        uc = self.unit_costs()
        g = lambda k, d=0.0: float(uc.get(k, d) or d)  # noqa: E731
        lines = []
        tot = 0.0

        def add(name: str, n: float, unit: float, est: bool = False):
            nonlocal tot
            usd = n * unit
            tot += usd
            lines.append(
                {
                    "item": name,
                    "n": round(n, 1),
                    "unit_usd": unit,
                    "usd": round(usd, 2),
                    "estimated": est,
                }
            )

        for p, v in res.items():
            if v["n_request"]:
                add(
                    f"resample {p} (generated samples)",
                    v["n_generated_samples"],
                    float(self.reg["pools"][p].get("usd_per_generated_sample") or 0),
                )
                if v["needs_precheck"]:
                    add(f"precheck {p}", 1, g("precheck_usd", 10.0), True)
        n_fill = fill["n_calls"] + fill["n_after_resample"]
        add("fill calls (sibfill-v1)", n_fill, g("fill_usd_per_call"))
        add("class calls (sibclass-v1, known new pairs)", len(new_pairs), g("class_usd_per_call"))
        # units without rung 1 yet (a new arm): rates from the current state (plan 0009).
        # A (said, item) pair already in the class map costs nothing; a new pair needs a class
        # call if one of its units reaches rung 2 (the plan-0009 rung-2 share).
        unknown = [u for u in units if u["unit"] not in rung1]
        r2 = g("rung2_share", 0.27)
        from . import classes as K

        cm = K.load_class_map(self.data) or {"pairs": {}}
        known = set(cm["pairs"])
        cand = {(u["said_norm"], u["id"]) for u in unknown} - {
            tuple(k.rsplit("|", 1)) for k in known
        }
        n_new_pairs = len(cand) * r2
        if unknown:
            add(
                "class calls (est. new pairs of new units)",
                n_new_pairs,
                g("class_usd_per_call"),
                True,
            )
            add("dedupe-incr call", 1, g("dedupe_usd_per_call"), True)
        st = self.state_rates()
        n_cross = len(cpl["crossing"]) + (n_new_pairs * st["crossing_per_pair"] if unknown else 0)
        add("concept calls (cap crossings)", n_cross, g("concepts_usd_per_call"), bool(unknown))
        n_cells = sum(len(v) for v in tpl.values()) + sum(
            2 * len(v) for v in cpl["appended"].values()
        )
        # a new pair in a kept class adds a concept when its said / gold is new to the class
        n_new_concepts = n_new_pairs * st["kept_share"] * st["new_concept_per_pair"]
        n_cells_est = n_cells + (2 * n_new_concepts + 3 * n_cross if unknown else 0)
        add("text cells (sibgen-chat-v2.1)", n_cells_est, g("gen_usd_per_cell"), bool(unknown))
        n_cap = sum(cap.values()) + (n_cells_est - n_cells) * g("texts_per_cell", 22.6)
        n_cap += n_cells * g("texts_per_cell", 22.6)
        add("capture prompts", n_cap, g("capture_usd_per_prompt"), bool(unknown))
        nc = max(1.0, st["n_classes"])

        def distinct(k: float) -> float:  # expected distinct classes hit by k draws
            return nc * (1 - (1 - 1 / nc) ** k)

        n_fit = len(refit) + (distinct(n_new_concepts) + n_cross if unknown else 0)
        n_r2_kept = len(unknown) * r2 * st["kept_share"]
        add("class refits", n_fit, g("fit_usd_per_class"), bool(unknown))
        n_sc = len(set(rescore)) + (max(n_fit, distinct(n_r2_kept)) if unknown else 0)
        add("class rescores", n_sc, g("score_usd_per_class"), bool(unknown))
        return {"lines": lines, "total_usd": round(tot, 2), "auto_usd": self.auto_usd}

    def state_rates(self) -> dict[str, float]:
        """Rates from the current classes / concepts: the share of rung-2 pairs in kept
        (>= 4-pair) classes, core concepts (said + gold) per kept pair, cap crossings per pair."""
        from . import classes as K

        classes = read_json(self.data / "classes.json", {"classes": {}})["classes"]
        concepts = read_json(self.data / "concepts.json", {"classes": {}})["classes"]
        n = max(1, len(classes))
        kept = self.kept_classes(classes)
        n_kept = sum(kept.values())
        core = sum(len(set(r["said"]) | set(r["gold"])) for r in concepts.values())
        n_cls = Counter(v["class"] for v in classes.values() if v["class"] != K.NO_CLASS)
        small = sum(1 for c, k in n_cls.items() if k == MIN_CLASS_PAIRS - 1)
        return {
            "kept_share": n_kept / n,
            "new_concept_per_pair": core / max(1, n_kept),
            "crossing_per_pair": small / n,
            "n_classes": float(len(concepts)),
        }

    # ------------------------------------------------------------ run
    def run(self) -> int:
        from . import fill_judge as F
        from . import ladder as L

        t0 = time.time()
        w = self.plan()
        self.print_plan(w)
        if w["estimate"]["total_usd"] > self.auto_usd and not self.yes:
            self.log(
                f"estimate ${w['estimate']['total_usd']:.2f} > ${self.auto_usd:.0f} "
                "($GATE_AUTO_USD): re-run with --yes"
            )
            self.record_pending("needs --yes", w)
            return EXIT_NEEDS_YES
        cal = read_json(self.out / "calibration.json")
        if cal is None or cal.get("prompt_version") != F.PROMPT_VERSION:
            self.log(
                f"WARNING: fill prompt {F.PROMPT_VERSION} differs from the calibrated "
                f"{(cal or {}).get('prompt_version')} (outputs/sibling_gate/calibration.json); "
                "calibration is not re-run by the gate"
            )
        tok = F.RETRY_FAILED.set(self.retry_failed)
        try:
            prev = read_json(self.out / "generation.json", {}) or {}
            prev_labels = self.load_prev_labels()
            if self.added:
                for a in self.added:
                    if a not in self.reg["gated_arms"]:
                        self.reg["gated_arms"].append(a)
                A.save(self.reg, self.reg_path)
                self.log(f"gated_arms += {self.added} ({self.reg_path})")
            units, uh = self.load_units()
            for a, h in uh.items():
                ph = ((prev.get("stamp") or {}).get("unit_hash") or {}).get(a)
                if ph and ph != h:
                    g0 = prev.get("generation")
                    self.log(f"[{a}] unit set changed since generation {g0}: re-gated")
            self.run_resample(units)
            rung1 = self.run_fill(units, self.resampled())
            pairs = self.class_pairs(units, rung1)
            classes = self.run_classes(pairs)
            concepts = self.run_concepts(classes)
            self.run_texts(units, classes, concepts)
            self.run_capture(concepts)
            self.run_fit(concepts)
            gen_next = int(prev.get("generation", -1)) + 1
            cps = L.load_class_probes(self.probes, concepts) if concepts else {}
            if self.run_freeze(units, rung1, classes, concepts, cps, gen_next):
                cps = L.load_class_probes(self.probes, concepts)
            if self.run_score(concepts, gen_next):
                cps = L.load_class_probes(self.probes, concepts)
            rows, nulls, ready = self.run_label(units, rung1, classes, concepts, cps)
            fit_hashes = {c: s["hash"] for c, s in self.fit_state(concepts).items()}
            st = self.stamp(uh, concepts, fit_hashes, rows)
            same = prev.get("stamp_sha") == sha256_json(st)
            gen = int(prev["generation"]) if same and "generation" in prev else gen_next
            ld = self.out / "labels"
            L.write_labels(rows, ld)
            write_json(ld / "matched_null.json", dict(nulls), indent=None)
            for a in self.gated:
                write_json(
                    U.stamp_path(a, ld),
                    {"generation": gen, "unit_hash": uh[a], "stamp_sha": sha256_json(st)},
                )
            cross = self.cross_digest(units, classes)
            changes = self.write_changes(prev_labels, rows, prev, units, classes, fit_hashes, cross)
            spend = {
                "llm_usd": round(self.spend_usd, 4),
                "llm_calls": self.llm.n_calls,
                "modal_launches": self.modal.n_launches,
            }
            gj = {
                "generation": gen,
                "stamp": st,
                "stamp_sha": sha256_json(st),
                "code_revision": code_revision(),
                "spend": spend,
                "estimate_usd": w["estimate"]["total_usd"],
                "wall_s": round(time.time() - t0, 1),
                "rung2_ready": ready,
                "n_units": len(rows),
                "pending": {},
                "changes": changes,
                "cross_digest": cross,
                "fit_hashes": fit_hashes,
                "ts": time.time(),
            }
            write_json(self.out / "generation.json", gj)
            self.log(
                f"generation {gen} ({'unchanged' if same else 'new'}): {len(rows)} units; "
                f"llm calls {self.llm.n_calls} (${self.spend_usd:.2f}); modal launches "
                f"{self.modal.n_launches}; wall {gj['wall_s']}s"
            )
            return EXIT_OK
        finally:
            F.RETRY_FAILED.reset(tok)

    def record_pending(self, why: str, w: Mapping[str, Any]) -> None:
        p = self.out / "generation.json"
        g = read_json(p)
        if g is None:
            return
        g["pending"] = {a: why for a in w["gated_arms"] if w["arms"][a]["changed"]}
        g["pending_estimate_usd"] = w["estimate"]["total_usd"]
        write_json(p, g)

    def load_prev_labels(self) -> dict[str, dict[str, Any]]:
        ld = self.out / "labels"
        out = {}
        if ld.exists():
            for f in ld.glob("*.jsonl"):
                for r in U.iter_jsonl(f):
                    out[r["unit"]] = r
        return out

    def cross_digest(self, units, classes) -> dict[str, str]:
        from . import classes as K
        from . import ladder as L

        idx = L.cross_index(units, K.canonical_map(classes))
        return {f"{i}|{c}": sha256_json(sorted(v))[:12] for (i, c), v in idx.items()}

    def write_changes(self, prev_rows, rows, prev, units, classes, fit_hashes, cur_cross):
        """``changes.md``: per previously gated arm, the units whose final changed and why
        (first match: class -> cross_method -> resample -> other) and the gated-share change."""
        from . import classes as K

        prev_arms = set((prev.get("stamp") or {}).get("gated_arms") or [])
        prev_fit = prev.get("fit_hashes") or {}
        prev_cross = prev.get("cross_digest") or {}
        canon = K.canonical_map(classes)
        unit_of = {u["unit"]: u for u in units}
        per_arm: dict[str, Counter] = defaultdict(Counter)
        share: dict[str, list[float]] = {}
        hall = U.HALLUCINATION_FINALS
        lines = ["# Sibling gate: what moved", ""]
        for a in sorted(prev_arms):
            old = [r for r in prev_rows.values() if r["arm"] == a]
            new = [r for r in rows if r["arm"] == a]
            if not old:
                continue
            share[a] = [
                sum(r.get("final") in hall for r in old) / max(1, len(old)),
                sum(r.get("final") in hall for r in new) / max(1, len(new)),
            ]
            for r in new:
                o = prev_rows.get(r["unit"])
                if o is None or o.get("final") == r.get("final"):
                    continue
                sn = unit_of[r["unit"]]["said_norm"]
                v = classes.get(f"{sn}|{r['id']}")
                cls = v["class"] if v else None
                ck = f"{r['id']}|{canon.get((sn, r['id']), sn)}"
                if cls and cls in fit_hashes and prev_fit.get(cls) != fit_hashes[cls]:
                    why = "class"
                elif prev_cross.get(ck) != cur_cross.get(ck):
                    why = "cross_method"
                elif o.get("counts") != r.get("counts"):
                    why = "resample"
                else:
                    why = "other"
                per_arm[a][why] += 1
        lines += [
            "| arm | changed units | class | cross_method | resample | other "
            "| gated share before -> after |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in sorted(share):
            c = per_arm[a]
            b, af = share[a]
            lines.append(
                f"| {a} | {sum(c.values())} | {c['class']} | {c['cross_method']} | {c['resample']} "
                f"| {c['other']} | {100 * b:.2f}% -> {100 * af:.2f}% |"
            )
        if not prev_cross:
            lines += [
                "",
                "(the previous generation recorded no cross-index digest: cross_method "
                "attribution falls through to the later rules)",
            ]
        (self.out / "changes.md").parent.mkdir(parents=True, exist_ok=True)
        (self.out / "changes.md").write_text("\n".join(lines) + "\n")
        return {a: dict(c) for a, c in per_arm.items()}

    # ------------------------------------------------------------ printing
    def print_plan(self, w: Mapping[str, Any]) -> None:
        P = print if not self.quiet else (lambda *a, **k: None)
        shown = self.show_arms or w["gated_arms"]
        P(
            f"== hbench gate --plan: gated set {len(w['gated_arms'])} arms"
            + (f" (adding {', '.join(w['added'])})" if w["added"] else "")
        )
        P(f"{'arm':22s} {'units':>7s} {'unit hash':>13s} {'prev':>13s}  changed")
        for a in shown:
            r = w["arms"].get(a)
            if r:
                P(
                    f"{a:22s} {r['n_units']:7d} {r['unit_hash']:>13s} {r['prev']!s:>13s}  "
                    f"{'yes' if r['changed'] else 'no'}"
                )
        P(
            f"\n{'pool':16s} {'keys':>6s} {'complete':>9s} {'request':>8s} {'exhausted':>9s} "
            f"{'launches':>8s} {'gen.cells':>9s} {'gen.samples':>11s}  precheck"
        )
        for p, v in w["resample"].items():
            P(
                f"{p:16s} {v['n_keys']:6d} {v['n_complete']:9d} {v['n_request']:8d} "
                f"{v['n_exhausted']:9d} {v['n_launches']:8d} {v['n_generated_cells']:9d} "
                f"{v['n_generated_samples']:11d}  {'needed' if v['needs_precheck'] else '-'}"
                + (f"  pending runs {v['pending_runs']}" if v["pending_runs"] else "")
            )
        f = w["fill"]
        kept = "retried" if self.retry_failed else "kept"
        P(
            f"\nfill calls: {f['n_calls']} now + {f['n_after_resample']} after resample "
            f"({f['n_failed_kept']} cached failures {kept})"
        )
        c = w["classes"]
        P(
            f"class pairs: {c['n_pairs']} known rung-2 pairs; new {c['new_pairs']}; "
            f"retry {c['retry_pairs']}; units without rung 1 yet: {w['n_units_without_rung1']}"
        )
        P(
            f"concepts: cap crossings {len(w['concepts']['crossing'])}; appended members "
            f"{sum(w['concepts']['appended'].values())} in {len(w['concepts']['appended'])} classes"
        )
        P(
            f"texts: {sum(w['texts'].values())} missing cells in {len(w['texts'])} classes; "
            f"capture prompts {sum(w['capture'].values())}; remerge {w['remerge'] or '-'}"
        )
        P(
            f"refits: {len(w['refit'])}; first distractor freezes: {w['first_freeze']}; "
            f"rescores: {len(w['rescore'])}"
        )
        fs = w["flip_sources"]
        P(
            f"flip sources: new arms {fs['new_arms'] or '-'}; existing units whose (item, said) "
            f"cross-index entries gain a new arm: {fs['cross_method_touched_units']}"
            + (f"; {fs['note']}" if fs["note"] else "")
        )
        e = w["estimate"]
        P("\ncost estimate (plan-0009 unit costs; est. = rate-based):")
        for ln in e["lines"]:
            if ln["n"]:
                P(
                    f"  {ln['item']:52s} {ln['n']:>10} x ${ln['unit_usd']:.5f} = ${ln['usd']:>8.2f}"
                    + ("  est." if ln["estimated"] else "")
                )
        P(
            f"  {'total':52s} {'':>10}   {'':9s}   ${e['total_usd']:>8.2f}  "
            f"(auto-proceed <= ${e['auto_usd']:.0f})"
        )
        P(f"\nzero work: {'yes' if w['zero_work'] else 'no'}")


# ---------------------------------------------------------------- label diff (§8)


def diff_labels(new_dir: Path, base_dir: Path, arms: Iterable[str]) -> dict[str, Any]:
    """Parsed rows keyed by unit: final, rung1, counts and the rung-2 trace must be equal."""
    fields = (
        "final",
        "rung1",
        "counts",
        "rung2",
        "branch",
        "sensitivity",
        "segments",
        "flags",
        "key",
    )
    out: dict[str, Any] = {"n_units": 0, "n_diff": 0, "per_arm": {}, "examples": []}
    for a in arms:
        pa, pb = Path(new_dir) / f"{a}.jsonl", Path(base_dir) / f"{a}.jsonl"
        ra = {r["unit"]: r for r in U.iter_jsonl(pa)} if pa.exists() else {}
        rb = {r["unit"]: r for r in U.iter_jsonl(pb)} if pb.exists() else {}
        d = Counter()
        for k in set(ra) | set(rb):
            if k not in ra or k not in rb:
                d["missing_new" if k not in ra else "missing_base"] += 1
                continue
            for f_ in fields:
                if ra[k].get(f_) != rb[k].get(f_):
                    d[f_] += 1
                    if len(out["examples"]) < 10:
                        out["examples"].append(
                            {"unit": k, "field": f_, "new": ra[k].get(f_), "base": rb[k].get(f_)}
                        )
        n_bad = sum(
            1
            for k in set(ra) | set(rb)
            if k not in ra or k not in rb or any(ra[k].get(f_) != rb[k].get(f_) for f_ in fields)
        )
        out["per_arm"][a] = {"n_units": len(rb), "n_diff_units": n_bad, "fields": dict(d)}
        out["n_units"] += len(rb)
        out["n_diff"] += n_bad
    return out


# ---------------------------------------------------------------- CLI


def auto_usd_from_env() -> float:
    raw = os.environ.get("GATE_AUTO_USD")
    if raw is None or raw.strip() == "":
        return GATE_AUTO_USD_DEFAULT
    try:
        v = float(raw)
    except ValueError:
        v = math.nan
    if not math.isfinite(v):
        raise GateExit(EXIT_FATAL, f"GATE_AUTO_USD={raw!r} is not a number (dollars, e.g. 300)")
    return v


def build_gate(a: argparse.Namespace, *, plan_only: bool) -> Gate:
    reg = A.load()
    added = [x for x in (a.add_arm or "").split(",") if x]
    try:
        reg = A.with_added(reg, added) if added else reg
    except A.RegistryError as e:
        raise GateExit(EXIT_FATAL, str(e)) from None
    return Gate(
        reg=reg,
        out=Path(a.out),
        data=Path(a.data_dir),
        cells_root=Path(a.cells_root),
        plan_only=plan_only,
        yes=a.yes,
        show_arms=[x for x in (a.arms or "").split(",") if x] or None,
        added=added,
        rung1_skip=a.rung1 == "skip",
        retry_failed=a.retry_failed,
        remerge=[x for x in (a.remerge or "").split(",") if x],
        allow_snapshot_drift=a.allow_snapshot_drift,
        capture_shards=a.capture_shards,
        resample_timeout_s=a.resample_timeout * 3600,
        auto_usd=auto_usd_from_env(),
        rpm=a.rpm,
        concurrency=a.concurrency,
        offline=a.offline,
    )


def cmd_gate(a: argparse.Namespace) -> int:
    try:
        if a.action == "import":
            from . import importer

            return importer.cmd_import(a)
        if a.action == "snapshot":
            reg = A.load()
            for name in reg["snapshots"]:
                s = make_snapshot(reg, name)
                print(f"snapshot {name}: {s['path']} tree {s['tree_sha256'][:16]}")
            A.save(reg)
            return EXIT_OK
        if a.action == "diff":
            res = diff_labels(Path(a.out) / "labels", Path(a.baseline), A.gated_arms())
            print(json.dumps(res, indent=1, ensure_ascii=False))
            return EXIT_OK if res["n_diff"] == 0 else 1
        g = build_gate(a, plan_only=a.plan)
        if a.plan:
            g.print_plan(g.plan())
            if a.plan_json:
                write_json(Path(a.plan_json), g.work)
            return EXIT_OK
        return g.run()
    except GateExit as e:
        print(f"GATE EXIT {e.code} ({EXIT_MEANING.get(e.code, '?')}): {e}", file=sys.stderr)
        return e.code
    except R.ModalFatal as e:
        print(f"GATE EXIT 3 (Modal): {e}", file=sys.stderr)
        return EXIT_FATAL
    except A.RegistryError as e:
        print(f"GATE EXIT 3 (registry): {e}", file=sys.stderr)
        return EXIT_FATAL


def add_parser(sub: Any) -> None:
    g = sub.add_parser("gate", help="plan 0010: the sibling gate as one pipeline step")
    g.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=("run", "import", "snapshot", "diff"),
        help="run (default) | import (plan-0009 state, no calls) | snapshot | diff",
    )
    g.add_argument("--plan", action="store_true", help="print the work + cost estimate only")
    g.add_argument("--plan-json", default=None, help="--plan: also write the plan as JSON")
    g.add_argument("--yes", action="store_true", help="proceed over $GATE_AUTO_USD (default 300)")
    g.add_argument("--arms", default=None, help="narrow what is printed (the gate set is fixed)")
    g.add_argument("--add-arm", default=None, help="append registered arm(s) to gated_arms")
    g.add_argument(
        "--rung1",
        choices=("run", "skip"),
        default="run",
        help="skip: a sampled arm without a pool is gated as deterministic",
    )
    g.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-ask cached failed LLM answers (default: they are final)",
    )
    g.add_argument("--remerge", default=None, help="comma list of classes to re-merge + recapture")
    g.add_argument("--allow-snapshot-drift", action="store_true")
    g.add_argument(
        "--offline",
        action="store_true",
        help="fail (exit 3) on any LLM call or Modal command (the idempotence check)",
    )
    g.add_argument("--resample-timeout", type=float, default=10.0, help="hours (default 10)")
    g.add_argument("--out", default=str(U.OUT), help="sibling-gate state root")
    g.add_argument("--data-dir", default=str(DATA_DIR))
    g.add_argument(
        "--cells-root",
        default=str(U.CELLS_ROOT_DEFAULT),
        help="DIR with <arm>/cells.jsonl for arms without a registered cells path",
    )
    g.add_argument("--rpm", type=float, default=1500.0)
    g.add_argument(
        "--capture-shards", type=int, default=0, help="capture containers (0 = auto, 1 per 1500)"
    )
    g.add_argument("--concurrency", type=int, default=64)
    g.add_argument("--from", dest="src", default=None, help="import: the plan-0009 worktree")
    g.add_argument("--baseline", default=None, help="diff: the baseline labels dir")
    g.set_defaults(fn=cmd_gate)
