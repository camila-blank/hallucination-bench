"""The ``sad`` eval (plan 0018): eval-vs-deployment awareness on SAD stages-oversight texts.

Its own driver and CLI (``hbench sad ...``, registered through ``cli.EVAL_CLIS``); it shares no
pipeline with the chat / story / AM evals. Steps: ``gate`` (Qwen3.6-27B's own belief, 10 samples
of SAD's question) -> ``select`` (balanced gated items, read sites) -> lens readouts of the PLAIN
prompt (Modal, global-workspace readers) -> ``detect`` (blind LLM detector of eval / deploy
beliefs in readouts) -> ``score`` / ``report``.

The SAD texts are contamination-protected: they live only under the gitignored
``data/sad/raw/`` (``scripts/sad/fetch_sad.py``); git holds ids, hashes, gate counts and site
positions, never a text or a token string."""

NAME = "sad"
