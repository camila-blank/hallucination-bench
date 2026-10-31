"""The only benign_roleplay code that touches WorkspaceBench (plan 0023 HB-A.3): a pin check and
subprocess calls of the pinned ``wsbench`` CLI. No WSB logic lives in HB: the loader, read
guards, digest, judge (jb-v3), post-processing and completeness rules all run inside WSB at
``WSB_SHA``. This module never reads, writes or exports an API key; the caller's environment
(e.g. ``ANTHROPIC_AUTH_TOKEN`` per command) passes through unchanged.

Layout of what is judged (``wsbench run ... reads=full``):

- ``hb_only`` ids, every arm: ``WSB_BANK_FILE_JAILBREAK_RECOGNITION=<hb_only bank>`` +
  ``layers=each`` into ``<out>/<arm>/hb_only`` (WSB writes ``jailbreak_recognition/all/`` and
  ``jailbreak_recognition/L<k>/``);
- WSB-shared ids: REUSED from the campaign's judged dirs (``<campaign>/s3d-<c>/`` for s3d-rl600,
  ``<campaign>/<arm>-all-full/`` for the jlens / mlnla-rl2-2000 all-layer unit), never re-judged;
  the jlens / mlnla-rl2-2000 single-layer configs are not in the campaign, so HB runs them
  (``layers=<k>`` into ``<out>/<arm>/shared/L<k>``)."""

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .bank import FAMILY, HB_ONLY_BANK

# The pinned WorkspaceBench sha the judge runs at: camila-blank/workspace-bench main 0b18789
# (post-audit: digest-v5, jb-v3, PRs #194-#199; #199 raised the jailbreak readout judge's output
# budget to 64k / 900 s after all-layer digest-v5 calls overflowed 16k). It is the sha the fix
# campaign re-judges jailbreak_recognition at, whose judged dirs HB reuses. Was 65951dc (#190,
# jb-v2 / digest-v4), then e8615c4, both on 2026-10-31. The jailbreak bank and its build record
# are byte-identical at all three shas (only evals/jailbreak_recognition/README.md changed), so
# items_v1 stays. A WSB fix PR bumps this one line.
WSB_SHA = "0b1878987927555a9d88f65c6fc3c14debb76e1e"
# The sha the frozen bank (items_v1.json / wsb_bank_hb_only_v1.json headers) was imported at:
# PR #188 head (plan 0037: frozen jailbreak bank v2 + its build/gate record). The bank and build
# files are byte-identical at WSB_SHA (test_bank_is_verbatim_copy reads them at WSB_SHA and
# checks the header sha256s), so a WSB_SHA bump that leaves them unchanged keeps items_v1.
BANK_WSB_SHA = "a2391b2b98eee55a43e659b7894cf752fe4d765c"
BANK_ENV = "WSB_BANK_FILE_" + FAMILY.upper()
GRID = list(range(20, 61, 4))  # WSB readplan.GRID (11 layers)
ARMS = ("s3d-rl600", "jlens", "mlnla-rl2-2000")
CAMPAIGN_PER_LAYER = {"s3d-rl600": "s3d"}  # arms whose single-layer configs the campaign judged
CAMPAIGN_ALL = {
    "s3d-rl600": "s3d-all",
    "jlens": "jlens-all-full",
    "mlnla-rl2-2000": "mlnla-rl2-2000-all-full",
}
# arms whose campaign judged dirs are nested per arm (``<campaign>/<arm>/all``,
# ``<campaign>/<arm>/L<k>``: the final O-lens campaign run, gw PR #646); every config is judged
# there, so HB reuses all of them. Opt-in (``--arms``), not in the default ARMS.
CAMPAIGN_NESTED = ("final-olens-rl600-x4",)
KNOWN_ARMS = (*ARMS, *CAMPAIGN_NESTED)


class WsbError(SystemExit):
    """Exit 2: a pin, layout or input problem (never a partial run)."""

    def __init__(self, msg: str) -> None:
        super().__init__(2)
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode:
        raise WsbError(f"git {' '.join(args)} failed in {root}: {r.stderr.strip()}")
    return r.stdout.strip()


def check_pin(root: Path, sha: str = WSB_SHA) -> None:
    """Exit 2 unless ``root`` is a clean WorkspaceBench checkout at ``sha``."""
    if not Path(root).is_dir():
        raise WsbError(f"no WorkspaceBench checkout at {root} (set HBENCH_WSBENCH_PUB)")
    head = _git(root, "rev-parse", "HEAD")
    if head != sha:
        raise WsbError(f"WorkspaceBench at {root} is {head}, not the pinned WSB_SHA {sha}")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise WsbError(f"WorkspaceBench at {root} has a dirty tree; pin needs a clean checkout")


def campaign_dir(campaign: Path, arm: str, unit: str) -> Path | None:
    """The reused campaign judged dir for (arm, unit) (``unit`` = ``all`` or ``L<k>``), or None
    when the campaign does not judge that config (HB judges it itself)."""
    if arm in CAMPAIGN_NESTED:
        return campaign / arm / unit
    if unit == "all":
        return campaign / CAMPAIGN_ALL[arm]
    if arm in CAMPAIGN_PER_LAYER:
        return campaign / f"{CAMPAIGN_PER_LAYER[arm]}-{unit[1:]}"
    return None


def result_path(d: Path, tag: str | None = None) -> Path:
    """``results.json`` of a ``wsbench run`` out dir (``tag`` = the layers=each sub-dir)."""
    return d / FAMILY / tag / "results.json" if tag else d / FAMILY / "results.json"


@dataclass
class Call:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    out: Path | None = None


def plan_calls(
    *,
    campaign: Path,
    readouts: Path,
    out: Path,
    layers: Sequence[int] | None = None,
    arms: Sequence[str] = ARMS,
    hb_bank: Path = HB_ONLY_BANK,
) -> list[Call]:
    """The ``wsbench run`` calls HB issues (exit 2 if a reused campaign config is missing)."""
    lay = sorted(set(layers)) if layers else list(GRID)
    if set(lay) - set(GRID):
        raise WsbError(f"layers {sorted(set(lay) - set(GRID))} are not in readplan.GRID {GRID}")
    units = ["all", *[f"L{k}" for k in lay]] if not layers or lay == GRID else []
    units = units or [f"L{k}" for k in lay]
    for arm in arms:
        for unit in units:
            d = campaign_dir(campaign, arm, unit)
            if d is not None and not result_path(d).exists():
                raise WsbError(f"campaign config missing for {arm} {unit}: {result_path(d)}")
    base = ["uv", "run", "--no-sync", "wsbench", "run", f"families={FAMILY}", "reads=full"]
    calls: list[Call] = []
    for arm in arms:
        for k in lay:
            if campaign_dir(campaign, arm, f"L{k}") is None:
                o = out / arm / "shared" / f"L{k}"
                calls.append(
                    Call(
                        [*base, f"readouts_root={readouts / arm}", f"layers={k}", f"out={o}"], out=o
                    )
                )
        env = {BANK_ENV: str(Path(hb_bank).resolve())}
        root = readouts / arm / "hb_only"
        if not layers or lay == GRID:
            o = out / arm / "hb_only"
            calls.append(Call([*base, f"readouts_root={root}", "layers=each", f"out={o}"], env, o))
        else:  # C3: a layer subset (the HB summary is then complete=False)
            for k in lay:
                o = out / arm / "hb_only" / f"L{k}"
                calls.append(
                    Call([*base, f"readouts_root={root}", f"layers={k}", f"out={o}"], env, o)
                )
    return calls


def run_calls(root: Path, calls: Sequence[Call], *, dry_run: bool = False) -> int:
    """Run each call in the pinned checkout (``check_pin`` first); the max exit code."""
    check_pin(root)
    code = 0
    for c in calls:
        pre = " ".join(f"{k}={v}" for k, v in c.env.items())
        print(f"[benign_roleplay] {pre} {' '.join(c.argv)}".strip(), flush=True)
        if dry_run:
            continue
        # a caller's WSB_BANK_FILE_* (e.g. the runbook's $HBENV) never leaks into a shared call
        base = {k: v for k, v in os.environ.items() if not k.startswith("WSB_BANK_FILE_")}
        r = subprocess.run(c.argv, cwd=root, env={**base, **c.env})
        code = max(code, r.returncode)
    return code


def bank_path(root: Path) -> Path:
    """``wsbench.registry.bank_path("jailbreak_recognition")`` in the pinned checkout (run there,
    with its own venv, the caller's ``WSB_BANK_FILE_*`` overrides stripped)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WSB_BANK_FILE_")}
    r = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            f"from wsbench.registry import bank_path; print(bank_path({FAMILY!r}))",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise WsbError(f"wsbench.registry.bank_path failed in {root}: {r.stderr.strip()}")
    return Path(r.stdout.strip().splitlines()[-1])
