"""Local DeepSeek-V4-Flash tokenizer helpers (the HF snapshot's tokenizer + its own
``encoding_dsv4.encode_messages``, the render every V4 producer uses) and the V4 window check.

The window check is plan 0003's ``bank.WindowChecker`` ported to the V4 template: a text's
``from_last_sentence_start`` window (wsbench readplan, V4 branch) must equal the request sentence
tokenised after one space plus the template tail (``<｜Assistant｜></think>``).
"""

from __future__ import annotations

import glob
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

WSB_SRC = os.environ.get("WSB_SRC", "/workspace/root/wsbench-pub-v4-opus/src")
SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/*"
)
SYSTEM_PROMPT = "You are a helpful assistant."


class V4:
    def __init__(self) -> None:
        snap = sorted(glob.glob(SNAP_GLOB))[-1]
        sys.path.insert(0, os.path.join(snap, "encoding"))
        sys.path.insert(0, WSB_SRC)
        from encoding_dsv4 import encode_messages  # type: ignore[import-not-found]
        from transformers import AutoTokenizer
        from wsbench.produce.render import display_tokens  # type: ignore[import-not-found]
        from wsbench.readplan import resolve  # type: ignore[import-not-found]

        self.tok = AutoTokenizer.from_pretrained(snap)
        self._enc, self._disp, self._resolve = encode_messages, display_tokens, resolve
        self._canon: dict[str, list[int]] = {}

    def ids(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        s = self._enc([dict(m) for m in messages], thinking_mode="chat")
        return list(self.tok.encode(s, add_special_tokens=False))

    def window(self, messages: Sequence[Mapping[str, str]]) -> tuple[list[int], list[int]]:
        """(positions, window ids)."""
        ids = self.ids(messages)
        pos = self._resolve({"kind": "from_last_sentence_start"}, self._disp(self.tok, ids))
        return pos, [ids[p] for p in pos]

    def canonical(self, request: str) -> list[int]:
        if request not in self._canon:
            ref = self.ids(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Hello. {request}"},
                ]
            )
            toks = self.tok.convert_ids_to_tokens(ref)
            a = max(i for i, t in enumerate(toks) if t == "<｜Assistant｜>")
            self._canon[request] = (
                list(self.tok.encode(" " + request, add_special_tokens=False)) + ref[a:]
            )
        return self._canon[request]

    def window_ok(self, messages: Sequence[Mapping[str, str]], request: str) -> bool:
        return self.window(messages)[1] == self.canonical(request)


def get() -> Any:
    return V4()
