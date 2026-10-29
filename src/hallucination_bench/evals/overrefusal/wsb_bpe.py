"""Port of WorkspaceBench ``src/wsbench/bpe.py`` @ ``1baaf00`` (verbatim below the docstring):
byte-level BPE vocabulary strings -> text for the plan-0020 digest bundles (J-lens cells)."""


def _bytes_to_unicode() -> dict[int, str]:
    """Returns the GPT-2 byte-level BPE table (Qwen's tokenizer uses the same): byte -> char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_U2B = {c: b for b, c in _bytes_to_unicode().items()}


def to_bytes(tok: str) -> bytes:
    """A byte-level BPE vocabulary string's bytes: byte-table chars back to their byte, anything
    else (e.g. the space that replaced ``Ġ``) to its own UTF-8 bytes."""
    return b"".join(bytes([_U2B[ch]]) if ch in _U2B else ch.encode("utf-8") for ch in tok)


def _control(text: str) -> bool:
    """A C0/C1 control character other than whitespace: what already-decoded text with a Latin
    Extended-A letter (``się``, ``děti``) turns into when decoded again. ``\r`` counts as
    whitespace: Qwen's vocabulary has real ``\r\n`` tokens (``čĊ``)."""
    return any(
        (ord(c) < 0x20 and c not in "\n\t\r\x0b\x0c") or 0x7F <= ord(c) <= 0x9F for c in text
    )


def decode_token(tok: str) -> str:
    """Decodes a byte-level BPE vocabulary string to text (``▁`` reads as a space). A string
    that does not decode cleanly is kept as written, with ``Ġ``/``Ċ`` rendered: bytes that are
    not valid UTF-8 (already-decoded Latin-1 text like ``café``, or a token holding part of a
    character, which therefore stays as its byte-level spelling) or that decode to a control
    character (already-decoded ``się``). Already-decoded text that happens to spell valid UTF-8
    in the byte table (``Ã©``) cannot be told apart and is decoded; producers should write the
    vocabulary strings (docs/producing_readouts.md)."""
    tok = tok.replace("▁", " ")
    try:
        text = to_bytes(tok).decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is None or _control(text):
        return tok.replace("Ġ", " ").replace("Ċ", "\n")
    return text
