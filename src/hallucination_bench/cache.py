"""Append-only JSONL call cache: rows are never rewritten; a changed fingerprint appends.

Ported from WorkspaceBench ``src/wsbench/cache.py`` @ 92d763e (MIT). Rows are
``{key, fingerprint, ts, result, meta}``; a row is *failed* iff ``row["result"] is None``. A
torn last line (an interrupted write) is terminated and skipped, so a re-run resumes safely.
"""

import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

_RESERVED = ("key", "fingerprint", "ts")


def fingerprint(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Cache:
    """Rows keyed by ``(key, fingerprint)``."""

    def __init__(self, path: Path, *, readonly: bool = False):
        self.path = Path(path)
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._n = 0
        torn = False
        if self.path.exists() and self.path.stat().st_size:
            data = self.path.read_bytes()
            torn = not data.endswith(b"\n")
            for line in data.decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("key"), str):
                    self._rows.setdefault(row["key"], []).append(row)
                    self._n += 1
        self._fh = None
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
            if torn:  # terminate a line torn by an interrupted write
                self._fh.write("\n")
                self._fh.flush()

    def get(self, key: str, fp: str, *, include_failed: bool = False) -> dict[str, Any] | None:
        """The newest row for ``(key, fp)``; failed rows only with ``include_failed``."""
        for row in reversed(self._rows.get(key, [])):
            if row.get("fingerprint") != fp:
                continue
            if include_failed or row.get("result") is not None:
                return row
        return None

    def put(self, key: str, fp: str, payload: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError(f"cache {self.path} is read-only")
        clash = [k for k in _RESERVED if k in payload]
        if clash:
            raise ValueError(f"payload uses reserved cache keys: {clash}")
        row = {"key": key, "fingerprint": fp, "ts": time.time(), **payload}
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._rows.setdefault(key, []).append(row)
        self._n += 1

    def keys(self) -> Iterator[str]:
        return iter(self._rows)

    def rows(self) -> Iterator[dict[str, Any]]:
        for rows in self._rows.values():
            yield from rows

    def __len__(self) -> int:
        return self._n

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
