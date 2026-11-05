"""Seed a hindex-v8 agentic_misalignment root from the v7 one (plan 0029 §3.1 / §3.3).

Per arm, copies the Opus detection rows (``{key}:D``, ``{key}:Dsum``, ``summ:{key}``,
``batchD:{sha}``, Stage H ``{key}:H:...`` and ``batchH:{sha}``) of ``<src>/<arm>/cache.jsonl``
into ``<dest>/<arm>/cache.jsonl``, and the model-level intent cache ``<src>/intent.jsonl`` into
``<dest>/intent.jsonl``. A plain filter: the rows keep their keys and fingerprints (the v8 run
uses ``--judge-model claude-opus-5-5 --map-model claude-sonnet-5-5``, so the detector
fingerprints match), and the v7 root is only read. No ``:M`` row is copied (the v8 map is
``claimmap-v2.8+am3``) unless ``--with-map`` (copying a pilot root's ``:M`` / ``sit:`` rows
into the real v8 root when the final prompt version equals the pilot's: same fingerprint, so no
cell is paid twice; plan §3.3).

    uv run --no-sync python scripts/v8/seed_am_root.py \\
        --src outputs/agentic_misalignment/claude-opus-5-5 \\
        --dest outputs/v8/agentic_misalignment/claude-opus-5-5 \\
        --arms s3d-rl600,mlnla-rl600-all11,jlens

The pilot root (``outputs/v8-pilot/...``) is seeded the same way.
"""

import argparse
import json
import shutil
from pathlib import Path

SEED_SUFFIXES = (":D", ":Dsum")
SEED_PREFIXES = ("summ:", "batchD:", "batchH:")
INTENT = "intent.jsonl"
CACHE = "cache.jsonl"
SITUATION_CACHE = "situation_cache.jsonl"


def is_seed_key(key: str) -> bool:
    """The detection-side rows of a v7 AM cache (never a claim-map or audit row)."""
    if key.startswith("audit:"):
        return False
    return key.endswith(SEED_SUFFIXES) or key.startswith(SEED_PREFIXES) or ":H:" in key


def is_map_key(key: str) -> bool:
    return key.endswith(":M")


def filter_cache(src: Path, dest: Path, *, with_map: bool = False, append: bool = False) -> int:
    """Copy the seed rows (plus ``:M`` rows with ``with_map``) of ``src`` to ``dest``."""
    if dest.exists() and not append:
        raise SystemExit(f"{dest} exists: seeding never overwrites (use --append to add rows)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as fi, dest.open("a", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            key = json.loads(line)["key"]
            if is_seed_key(key) or (with_map and is_map_key(key)):
                fo.write(line if line.endswith("\n") else line + "\n")
                n += 1
    return n


def seed(
    src: Path, dest: Path, arms: list[str], *, with_map: bool = False, append: bool = False
) -> dict[str, int]:
    if src.resolve() == dest.resolve():
        raise SystemExit("--src and --dest are the same root")
    out: dict[str, int] = {}
    for arm in arms:
        out[arm] = filter_cache(
            src / arm / CACHE, dest / arm / CACHE, with_map=with_map, append=append
        )
        if with_map and (src / arm / SITUATION_CACHE).is_file():
            sit_dest = dest / arm / SITUATION_CACHE
            if sit_dest.exists() and not append:
                raise SystemExit(f"{sit_dest} exists")
            with sit_dest.open("a", encoding="utf-8") as fo:
                fo.write((src / arm / SITUATION_CACHE).read_text(encoding="utf-8"))
    di = dest / INTENT
    if (src / INTENT).is_file() and not di.exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / INTENT, di)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="v7 AM model root (read only)")
    ap.add_argument("--dest", required=True, type=Path, help="v8 (or v8-pilot) AM model root")
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--with-map", action="store_true", help="also copy :M / sit: rows (pilot)")
    ap.add_argument("--append", action="store_true", help="add to an existing dest cache")
    a = ap.parse_args()
    res = seed(a.src, a.dest, a.arms.split(","), with_map=a.with_map, append=a.append)
    print(json.dumps({"dest": str(a.dest), "rows": res}, indent=1))


if __name__ == "__main__":
    main()
