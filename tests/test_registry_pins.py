"""Plan 0029 §3.1 / §6: the claimmap-v2.7 (+am2) prompts and schemas are pinned to the hashes
recorded from main BEFORE the v2.8 change (merge base 4034b31), and the registry serves them
unchanged for every v3-v7 root."""

import hashlib
import json

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges import claimmap_am as cam
from hallucination_bench.judges import claimmap_v2 as cm

# recorded from the unmodified code (main @ 4034b31) before any plan-0029 edit
PINS = {
    "SYSTEM": "1d5ecd60f7eb6628fdcfbda7cc2d710a808b321e239f8bb6e2cdcacd65e661cb",
    "USER_TEMPLATE": "40275aba50edbe765ea1617b2846f99f67b5b7a6ecc1db86f8bc204d30390d51",
    "SYSTEM_V5C": "2a626bea35291aa6fa45d9ea8c5559b8d3788ea2dc497df692a1923ef7053aa6",
    "USER_TEMPLATE_V5C": "83aaa8c50cd52b3ff33a989874719467179be23f5737df0a5a2ab3050b3b2824",
    "SCHEMA_none": "ae6554164d594acdf5b3c78420a3ab0563a60a449852b1e1adea687a858cb6e1",
    "SCHEMA_v5c": "55de26fe713e89927df8a58b1b75b720878e8b23721446feec2eecc42e4a9655",
    "AM_SYSTEM": "1d5ecd60f7eb6628fdcfbda7cc2d710a808b321e239f8bb6e2cdcacd65e661cb",
    "AM_USER_TEMPLATE": "d615ac4d64d13f9f7deb116796fbf26a6bdd89f1ce884bd737cfeea789248b7b",
    "AM_SCHEMA": "ae6554164d594acdf5b3c78420a3ab0563a60a449852b1e1adea687a858cb6e1",
    "RENDER": "5fe7f5c47f9834e5cc5b0c1f4369e4effec2b20ac5e50dd07609430a1864fe1f",
    "RENDER_ANCHORED": "c562a1746b67c36f147c85f0a68b67a767791837695ab9752066f31fcdf75387",
}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def hs(schema: dict) -> str:
    return h(json.dumps(schema, sort_keys=True))


def test_module_constants_keep_their_v27_values():
    assert cm.PROMPT_VERSION == "claimmap-v2.7" and cam.PROMPT_VERSION == "claimmap-v2.7+am2"
    assert h(cm.SYSTEM) == PINS["SYSTEM"]
    assert h(cm.USER_TEMPLATE) == PINS["USER_TEMPLATE"]
    assert h(cm.SYSTEM_V5C) == PINS["SYSTEM_V5C"]
    assert h(cm.USER_TEMPLATE_V5C) == PINS["USER_TEMPLATE_V5C"]
    assert hs(cm.SCHEMA) == PINS["SCHEMA_none"] and hs(cm.SCHEMA_V5C) == PINS["SCHEMA_v5c"]
    assert hs(cm.schema_for("none")) == PINS["SCHEMA_none"]
    assert hs(cm.schema_for("v5c")) == PINS["SCHEMA_v5c"]
    assert (
        h(cam.SYSTEM) == PINS["AM_SYSTEM"] and h(cam.USER_TEMPLATE_AM) == PINS["AM_USER_TEMPLATE"]
    )
    assert hs(cam.SCHEMA) == PINS["AM_SCHEMA"]
    assert (
        h(tx.render()) == PINS["RENDER"] and h(tx.render(anchored=True)) == PINS["RENDER_ANCHORED"]
    )


def test_registry_v27_specs_equal_the_pins():
    s = cm.VERSIONS["claimmap-v2.7"]
    assert h(s.system) == PINS["SYSTEM"] and h(s.user_template) == PINS["USER_TEMPLATE"]
    assert h(s.system_v5c) == PINS["SYSTEM_V5C"]
    assert h(s.user_template_v5c) == PINS["USER_TEMPLATE_V5C"]
    assert hs(cm.schema_for("none", "claimmap-v2.7")) == PINS["SCHEMA_none"]
    assert hs(cm.schema_for("v5c", "claimmap-v2.7")) == PINS["SCHEMA_v5c"]
    assert s.taxonomy_version == "v2-r17" and s.scoring_version == "v2-r20"
    a = cam.VERSIONS["claimmap-v2.7+am2"]
    assert h(a.system) == PINS["AM_SYSTEM"] and h(a.user_template) == PINS["AM_USER_TEMPLATE"]
    assert hs(a.schema) == PINS["AM_SCHEMA"]
    assert h(tx.render(version="claimmap-v2.7")) == PINS["RENDER"]
    assert h(tx.render(anchored=True, version="claimmap-v2.7")) == PINS["RENDER_ANCHORED"]


def test_v28_specs_differ_and_are_registered():
    assert set(cm.VERSIONS) >= {"claimmap-v2.7", "claimmap-v2.8"}
    assert set(cam.VERSIONS) >= {"claimmap-v2.7+am2", "claimmap-v2.8+am3"}
    s8 = cm.VERSIONS["claimmap-v2.8"]
    assert h(s8.system) != PINS["SYSTEM"] and h(s8.user_template) != PINS["USER_TEMPLATE"]
    assert hs(cm.schema_for("none", "claimmap-v2.8")) != PINS["SCHEMA_none"]
    assert s8.taxonomy_version == "v2-r21" and s8.scoring_version == "v2-r21"
    a8 = cam.VERSIONS["claimmap-v2.8+am3"]
    assert a8.system == s8.system and a8.chat_version == "claimmap-v2.8"
    assert hs(a8.schema) == hs(cm.schema_for("none", "claimmap-v2.8"))
    # v2.8.1 is registered only if the tag gate forces it (plan §3.3), in a follow-up PR
    assert "claimmap-v2.8.1" not in cm.VERSIONS


# Plan 0029 amendment (claimmap-v2.9): the claimmap-v2.8 (+am3) prompts and schemas are pinned
# to the hashes recorded from main @ 6c58a28 BEFORE the v2.9 change, so pilot2's v2.8 roots
# replay byte-identical prompts and fingerprints.
PINS_V28 = {
    "V28_SYSTEM": "c100c26b4100f5ff8c1f02abf85c54dbc985d58055877f4dd5e684baa1c7b480",
    "V28_USER_TEMPLATE": "e5462dd8fc7c858eb8850191e379d25fc59451290de6bdc9a256f5fa6617a928",
    "V28_SYSTEM_V5C": "a9992b3f5565974da57e1659a699d07cb486e3f1302b73cbd2e4f56a2448fce8",
    "V28_USER_TEMPLATE_V5C": "1d293425fd3b23928fa32404fa641ab05a0ba0b960b9cb7227fe2b231d4f58dd",
    "V28_SCHEMA_none": "20f821c661d18f85df2a5a3f0f6668e797507332a0729b2c226a9b666408e12c",
    "V28_SCHEMA_v5c": "5810fccc19203ae95ab9d895161645f95a56f881f5f416c8011856dab82952f0",
    "V28_AM_USER_TEMPLATE": "855b58cff072601ba9e332473e4329544eb748d5de1cd71b88a04e16af9306c7",
    "V28_RENDER": "c3c77ded5a6d00fada6611f1438b150256e37844f2cdcfe24a2db04fe791bcf9",
    "V28_RENDER_ANCHORED": "4d5bd7674657d40c057aed1f45841135b8fbf33c3d2f2ebe98f31f3b1bb49d19",
}


def test_registry_v28_specs_equal_the_v28_pins():
    s = cm.VERSIONS["claimmap-v2.8"]
    assert h(s.system) == PINS_V28["V28_SYSTEM"]
    assert h(s.user_template) == PINS_V28["V28_USER_TEMPLATE"]
    assert h(s.system_v5c) == PINS_V28["V28_SYSTEM_V5C"]
    assert h(s.user_template_v5c) == PINS_V28["V28_USER_TEMPLATE_V5C"]
    assert hs(cm.schema_for("none", "claimmap-v2.8")) == PINS_V28["V28_SCHEMA_none"]
    assert hs(cm.schema_for("v5c", "claimmap-v2.8")) == PINS_V28["V28_SCHEMA_v5c"]
    assert h(tx.render(version="claimmap-v2.8")) == PINS_V28["V28_RENDER"]
    assert h(tx.render(anchored=True, version="claimmap-v2.8")) == PINS_V28["V28_RENDER_ANCHORED"]
    a = cam.VERSIONS["claimmap-v2.8+am3"]
    assert h(a.system) == PINS_V28["V28_SYSTEM"]
    assert h(a.user_template) == PINS_V28["V28_AM_USER_TEMPLATE"]
    assert hs(a.schema) == PINS_V28["V28_SCHEMA_none"]


def test_v29_specs_differ_from_the_v28_pins():
    s9 = cm.VERSIONS["claimmap-v2.9"]
    assert h(s9.system) != PINS_V28["V28_SYSTEM"]
    assert h(s9.user_template) != PINS_V28["V28_USER_TEMPLATE"]
    assert hs(cm.schema_for("none", "claimmap-v2.9")) != PINS_V28["V28_SCHEMA_none"]
    a9 = cam.VERSIONS["claimmap-v2.9+am3"]
    assert a9.system == s9.system and a9.chat_version == "claimmap-v2.9"
    assert h(a9.user_template) != PINS_V28["V28_AM_USER_TEMPLATE"]
