"""sibling_probe: are same-category sibling mentions in lens readouts hallucinations?

The final sibling eval is the user-modelling ladder (plans 0006 + 0011; three domains: country,
capital_city = the city the user is visiting, holiday): ``um_ladder`` / ``um_domains`` /
``um_probe`` / ``um_regions``, driven by ``scripts/sibling_probe/um_ladder.py``. Shared pieces it
builds on, from the superseded plan 0003 / 0005 evals (removed from the tree in plan 0013): the
concept inventory and alias matcher (``concepts``), the bank / ReadSpec writers / leak, 8-gram and
window-identity checks (``bank``), cross-fit DiffMean probes (``probes``, run on Modal by
``scripts/sibling_probe/um_probes_modal.py``) and the lexicon matcher (``ladder_lex``).
"""

NAME = "sibling_probe"
CATEGORIES = ("country", "occupation", "emotion", "farm_animal", "capital_city", "tree", "holiday")
SYSTEM_PROMPT = "You are a helpful assistant."
