"""sibling_probe: are same-category sibling mentions in lens readouts hallucinations?

Plan ``plans/0003-sibling-concept-probes.md``. Stage 1 (this package so far): the concept
inventory and alias matcher (``concepts``), and the bank / ReadSpec writers / leak, 8-gram and
window-identity checks (``bank``).
"""

NAME = "sibling_probe"
CATEGORIES = ("country", "occupation", "emotion", "farm_animal", "capital_city", "tree", "holiday")
SYSTEM_PROMPT = "You are a helpful assistant."
