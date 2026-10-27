"""Fetch the SAD stages-oversight texts (plan 0018 §Bank). Deterministic and re-runnable.

Downloads ``sad/stages/oversight/structs.zip`` from LRudL/sad at the pinned commit, checks its
sha256, unzips the two batches (``test_oversight`` -> provenance eval, ``deploy_oversight`` ->
deploy) with the password SAD publishes in its ``unzip.sh``, and writes

* ``data/sad/raw/items.jsonl`` -- the texts. GITIGNORED: SAD asks that its question texts never
  appear in plain text anywhere scrapable (this repo is public). Never commit this file.
* ``data/sad/raw/templates.json`` -- SAD's question variants (also zipped upstream; gitignored).
* ``data/sad/manifest.json`` -- ids, provenance, source, sha256 and length only (committed).

    uv run --no-sync python scripts/sad/fetch_sad.py            # download + unpack + manifest
    uv run --no-sync python scripts/sad/fetch_sad.py --check    # re-derive, compare to manifest
"""

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter

import yaml

from hallucination_bench.evals.sad import bank

ZIP_SHA256 = "5b620b7bf45d04a2f79b1c7d2e825069fe774d853beeaff11b8b43c442c50ebb"
# published in LRudL/sad unzip.sh (PASSWORD=...): the archive is a contamination guard, not a secret
PASSWORD = b"sadtimesforthesetimes"
URL = f"https://github.com/{bank.SAD_REPO}/raw/{bank.SAD_COMMIT}/{bank.SAD_PATH}"


def fetch_zip() -> bytes:
    cached = bank.raw_dir() / "structs.zip"
    if cached.exists():
        data = cached.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as r:
            data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != ZIP_SHA256:
        raise SystemExit(f"structs.zip sha256 {got} != pinned {ZIP_SHA256}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def build_rows(data: bytes) -> list[dict]:
    z = zipfile.ZipFile(io.BytesIO(data))
    z.setpassword(PASSWORD)
    rows: list[dict] = []
    seen: set[str] = set()
    for batch, prov in bank.BATCHES.items():
        doc = yaml.safe_load(z.read(f"batch/{batch}.yaml"))
        for s in doc["samples"]:
            if s["id"] in seen:  # one WildChat text is listed twice in deploy_oversight: keep one
                continue
            seen.add(s["id"])
            right = s["choices_right"]
            # every stages-oversight question is phrased so "yes" = testing/evaluation
            want = ["yes"] if prov == "eval" else ["no"]
            if right != want:
                raise SystemExit(f"{s['id']}: choices_right {right} != {want} for {batch}")
            rows.append(
                {
                    "id": bank.short_id(s["id"]),
                    "sad_id": s["id"],
                    "provenance": prov,
                    "source": s["splits"]["source"],
                    "text": s["body"],
                }
            )
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("short ids collide; lengthen bank.short_id")
    return rows


def manifest(rows: list[dict]) -> dict:
    return {
        "meta": {
            "sad_repo": bank.SAD_REPO,
            "sad_commit": bank.SAD_COMMIT,
            "sad_path": bank.SAD_PATH,
            "zip_sha256": ZIP_SHA256,
            "n_items": len(rows),
            "n_duplicates_dropped": 1,  # one deploy_oversight sample appears twice (same id/text)
            "by_provenance": dict(Counter(r["provenance"] for r in rows)),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "note": "texts are NOT in git (contamination guard); data/sad/raw/items.jsonl",
        },
        "items": [
            {
                "id": r["id"],
                "sad_id": r["sad_id"],
                "provenance": r["provenance"],
                "source": r["source"],
                "sha256": bank.sha256_text(r["text"]),
                "n_chars": len(r["text"]),
            }
            for r in rows
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare to the committed manifest")
    args = ap.parse_args()
    rows = build_rows(fetch_zip())
    man = manifest(rows)
    if args.check:
        ok = json.loads(bank.MANIFEST.read_text()) == man
        print("manifest matches" if ok else "MANIFEST DIFFERS")
        return 0 if ok else 1
    out = bank.raw_dir() / "items.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = yaml.safe_load(
        zipfile.ZipFile(io.BytesIO(fetch_zip())).read("sampletemplates/default.yaml", pwd=PASSWORD)
    )["define_your_special_variables"]
    (bank.raw_dir() / "templates.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    bank.MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(rows)} items, gitignored) and {bank.MANIFEST}: {man['meta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
