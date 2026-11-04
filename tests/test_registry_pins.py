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
