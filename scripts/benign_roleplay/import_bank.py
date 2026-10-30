"""Import the benign_roleplay bank from a pinned WorkspaceBench checkout (plan 0023 HB-A.1).

No sampler and no gate judge: the gate records come from WSB's build_v2.json (plan 0037 J5).

    HBENCH_WSBENCH_PUB=/path/to/workspace-bench uv run --no-sync python \\
        scripts/benign_roleplay/import_bank.py [--n-target 150]
"""

import sys

from hallucination_bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benign-roleplay", "import", *sys.argv[1:]]))
