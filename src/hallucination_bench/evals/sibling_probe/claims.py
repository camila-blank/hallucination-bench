"""Plan 0004: sibling / substitution claim units from judge cells, and the post hoc consensus.

A **claim unit** is (method, set, item, layer, pos, concept), deduplicated across segments; it
keeps the segment texts. Extraction is the same for every method (plan rev 2 item 4):

* **set A** (plan-0003 sibling bank): a sibling mention (inventory alias hit of a concept that is
  not the gold, not named in the prompt, not named in the response) in a segment whose verdict
  is not-correct (plan 0003 M = {hallucination, probe_candidate, unresolved, junk}), or a
  substitution-leaf segment whose ``probe_target`` names such a concept. J-lens: the plan-0003
  token+summary rule (the concept must also be a raw top-10 token alias hit).
* **set B** (basic_readout): a substitution-leaf segment, its concept by ``brnorm-v1``
  (``member`` / ``out_of_inventory`` / ``entity_substitution``), or an M-verdict segment with an
  inventory alias hit; the concept is not the gold and not named in the prompt or response.
  J-lens additionally needs a raw top-10 token alias hit of the concept (entity substitutions
  excepted: they need only the summary).

**Consensus** (plan rev 4 item 1), per set: a triple (item, pos, concept) where s3d and J-lens
claim it at L_s, L_j with |L_s - L_j| <= 4 and NLA-RL L42 claims it at that (item, pos). A unit
at the triple's (item, pos, concept) is ``consensus`` if it is an NLA-RL L42 unit or its layer
lies in [min(L_s, L_j) - 4, max(L_s, L_j) + 4] for some qualifying (L_s, L_j). The ML-NLA
variant ``consensus_mlnla``: s3d, J-lens and ML-NLA at layers pairwise within 4; an ML-NLA unit
gets it if its own layer is part of such a triple; an s3d / J-lens unit if its layer lies in
[min - 4, max + 4] of one.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from hallucination_bench import taxonomy_v2 as tx

M_VERDICTS = frozenset({tx.V_HALLUCINATION, tx.V_PROBE, tx.V_UNRESOLVED, tx.V_JUNK})
SUB_LEAVES = frozenset(tx.SUBSTITUTION_LEAVES)
GRID = 4  # one grid step (layers 20, 24, ..., 60)

# method names (the figures / tables use these)
S3D, JLENS, NLA42, MLNLA = "s3d", "jlens", "nla-rl-L42", "mlnla-rl600"
METHODS: tuple[str, ...] = (S3D, NLA42, JLENS, MLNLA)
PRIMARY: tuple[str, ...] = (S3D, NLA42, JLENS)
METHOD_LABEL = {
    S3D: "s3d O-lens",
    NLA42: "NLA-RL L42",
    JLENS: "J-lens",
    MLNLA: "ML-NLA RL600",
}
# judge arm directory -> method, per set
ARM_METHOD_A = {
    "s3d-rl600-all11": S3D,
    "jlens-all11": JLENS,
    "nla-rl-L42": NLA42,
    "mlnla-rl600-all11": MLNLA,
}
ARM_METHOD_B = {
    "br-s3d-rl600": S3D,
    "br-jlens": JLENS,
    "br-nla-rl-L42": NLA42,
    "br-mlnla-rl600": MLNLA,
}


@dataclass
class Unit:
    method: str
    set: str  # "A" | "B"
    item: str
    layer: int
    pos: int
    concept: str
    category: str
    gold: str
    kind: str = "sibling"  # "sibling" | "out_of_inventory" | "entity_substitution"
    segments: list[dict[str, Any]] = field(default_factory=list)
    readouts: list[int] = field(default_factory=list)
    leaves: list[str] = field(default_factory=list)
    cell_key: str = ""

    @property
    def uid(self) -> str:
        return f"{self.set}|{self.method}|{self.item}|{self.layer}|{self.pos}|{self.concept}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["uid"] = self.uid
        return d


def _add(units: dict[str, Unit], u: Unit, seg_i: int, seg: Mapping[str, Any], readout: int) -> None:
    cur = units.setdefault(u.uid, u)
    if all(s.get("i") != seg_i for s in cur.segments):
        cur.segments.append(
            {
                "i": seg_i,
                "readout": int(seg["readout"]),
                "start": int(seg["start"]),
                "end": int(seg["end"]),
                "text": seg["text"],
                "leaf": seg.get("leaf"),
                "verdict": seg.get("verdict"),
                "probe_target": seg.get("probe_target"),
            }
        )
        cur.segments.sort(key=lambda s: (s["readout"], s["start"]))
    if readout not in cur.readouts:
        cur.readouts.append(readout)
        cur.readouts.sort()
    if seg.get("leaf") and seg["leaf"] not in cur.leaves:
        cur.leaves.append(seg["leaf"])


def _excluded(rec: Mapping[str, Any], concept: str) -> bool:
    return (
        concept == rec["gold"]
        or concept in set(rec.get("named") or ())
        or concept in set(rec.get("response_named") or ())
    )


def units_set_a(method: str, cells: Iterable[Mapping[str, Any]]) -> list[Unit]:
    """Claim units of one set-A arm from its judge ``cells.jsonl`` records."""
    units: dict[str, Unit] = {}
    for rec in cells:
        if rec.get("status") != "judged":
            continue
        segs = rec.get("segments") or []
        tokens = rec.get("kind") == "tokens"
        th = set(rec.get("token_hits") or ())

        def mk(c: str, rec: Mapping[str, Any] = rec) -> Unit:
            return Unit(
                method,
                "A",
                rec["id"],
                int(rec["layer"]),
                int(rec["pos"]),
                c,
                rec["category"],
                rec["gold"],
                cell_key=rec["key"],
            )

        for m in rec.get("mentions") or ():
            c = m["concept"]
            i = m.get("segment")
            if i is None or _excluded(rec, c) or m.get("verdict") not in M_VERDICTS:
                continue
            if tokens and c not in th:
                continue
            _add(units, mk(c), i, segs[i], int(segs[i]["readout"]))
        for i, s in enumerate(segs):
            if s.get("leaf") not in SUB_LEAVES:
                continue
            for c in s.get("target_concepts") or ():
                if _excluded(rec, c) or (tokens and c not in th):
                    continue
                _add(units, mk(c), i, s, int(s["readout"]))
    return sorted(units.values(), key=lambda u: (u.item, u.layer, u.pos, u.concept))


NormFn = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[str, str] | None]


def units_set_b(
    method: str,
    cells: Iterable[Mapping[str, Any]],
    norm: NormFn,
    token_concepts: Callable[[Mapping[str, Any]], set[str]] | None = None,
) -> list[Unit]:
    """Claim units of one set-B arm. ``norm(rec, segment)`` -> (kind, concept) for a
    substitution-leaf segment (the cached ``brnorm-v1`` answer; kinds ``member``,
    ``out_of_inventory``, ``entity_substitution``) or None (``none`` / not cached).
    ``token_concepts(rec)``: the inventory concepts hit by the J-lens raw top-10 tokens."""
    units: dict[str, Unit] = {}
    for rec in cells:
        if rec.get("status") != "judged":
            continue
        segs = rec.get("segments") or []
        tokens = rec.get("kind") == "tokens"
        th = (
            token_concepts(rec) if (tokens and token_concepts) else set(rec.get("token_hits") or ())
        )

        def mk(c: str, kind: str, rec: Mapping[str, Any] = rec) -> Unit:
            return Unit(
                method,
                "B",
                rec["id"],
                int(rec["layer"]),
                int(rec["pos"]),
                c,
                rec["category"],
                rec["gold"],
                kind=kind,
                cell_key=rec["key"],
            )

        for m in rec.get("mentions") or ():
            c = m["concept"]
            i = m.get("segment")
            if i is None or _excluded(rec, c) or m.get("verdict") not in M_VERDICTS:
                continue
            if tokens and c not in th:
                continue
            _add(units, mk(c, "sibling"), i, segs[i], int(segs[i]["readout"]))
        for i, s in enumerate(segs):
            if s.get("leaf") not in SUB_LEAVES:
                continue
            r = norm(rec, s)
            if r is None:
                continue
            kind, c = r
            if kind == "member":
                if _excluded(rec, c) or (tokens and c not in th):
                    continue
                _add(units, mk(c, "sibling"), i, s, int(s["readout"]))
            elif kind == "out_of_inventory":
                if c == rec["gold"] or c in set(rec.get("named") or ()):
                    continue
                if c in set(rec.get("response_named") or ()):
                    continue
                _add(units, mk(c, "out_of_inventory"), i, s, int(s["readout"]))
            elif kind == "entity_substitution":
                _add(units, mk(c, "entity_substitution"), i, s, int(s["readout"]))
    return sorted(units.values(), key=lambda u: (u.item, u.layer, u.pos, u.concept))


# ------------------------------------------------------------------ consensus


def _layers_by(units: Iterable[Unit], method: str) -> dict[tuple[str, str, int, str], set[int]]:
    out: dict[tuple[str, str, int, str], set[int]] = defaultdict(set)
    for u in units:
        if u.method == method and u.kind == "sibling":
            out[(u.set, u.item, u.pos, u.concept)].add(u.layer)
    return out


def consensus_triples(units: Sequence[Unit]) -> dict[tuple[str, str, int, str], list[tuple]]:
    """(set, item, pos, concept) -> the qualifying (L_s, L_j) pairs (NLA-RL L42 present)."""
    s3d, jl, nla = (_layers_by(units, m) for m in (S3D, JLENS, NLA42))
    out: dict[tuple[str, str, int, str], list[tuple]] = {}
    for k, ls in s3d.items():
        if k not in nla or k not in jl:
            continue
        pairs = [(a, b) for a in sorted(ls) for b in sorted(jl[k]) if abs(a - b) <= GRID]
        if pairs:
            out[k] = pairs
    return out


def consensus_mlnla_triples(
    units: Sequence[Unit],
) -> dict[tuple[str, str, int, str], list[tuple]]:
    """(set, item, pos, concept) -> the (L_s, L_j, L_m) triples, pairwise within one step."""
    s3d, jl, ml = (_layers_by(units, m) for m in (S3D, JLENS, MLNLA))
    out: dict[tuple[str, str, int, str], list[tuple]] = {}
    for k, ls in s3d.items():
        if k not in jl or k not in ml:
            continue
        tr = [
            (a, b, c)
            for a in sorted(ls)
            for b in sorted(jl[k])
            for c in sorted(ml[k])
            if abs(a - b) <= GRID and abs(a - c) <= GRID and abs(b - c) <= GRID
        ]
        if tr:
            out[k] = tr
    return out


def consensus_flags(units: Sequence[Unit]) -> dict[str, dict[str, bool]]:
    """uid -> {"consensus": bool, "consensus_mlnla": bool} (plan rev 4 item 1)."""
    tri = consensus_triples(units)
    trm = consensus_mlnla_triples(units)
    out: dict[str, dict[str, bool]] = {}
    for u in units:
        k = (u.set, u.item, u.pos, u.concept)
        c = False
        if u.kind == "sibling" and k in tri:
            if u.method == NLA42:
                c = True
            elif u.method in (S3D, JLENS):
                c = any(min(a, b) - GRID <= u.layer <= max(a, b) + GRID for a, b in tri[k])
        cm = False
        if u.kind == "sibling" and k in trm:
            if u.method == MLNLA:
                cm = any(u.layer == t[2] for t in trm[k])
            elif u.method in (S3D, JLENS):
                cm = any(min(t) - GRID <= u.layer <= max(t) + GRID for t in trm[k])
        out[u.uid] = {"consensus": c, "consensus_mlnla": cm}
    return out
