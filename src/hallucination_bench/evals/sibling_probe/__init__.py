"""sibling_probe: are same-category sibling mentions in lens readouts hallucinations?

Plan ``plans/0003-sibling-concept-probes.md``. Stage 1: the concept inventory and alias matcher
(``concepts``), the bank / ReadSpec writers /
leak, 8-gram and window-identity checks (``bank``). Stage 2: cross-fit DiffMean probes
(``probes``, run on Modal via ``probe_io`` + ``scripts/sibling_probe/probes_modal.py``), the
judge (``judge``), scoring and the verdict (``score``), and the report (``report``).
"""

NAME = "sibling_probe"
CATEGORIES = ("country", "occupation", "emotion", "farm_animal", "capital_city", "tree", "holiday")
SYSTEM_PROMPT = "You are a helpful assistant."
