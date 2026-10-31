"""``benign_roleplay`` eval (plan 0023): on harmless role-play prompts that Qwen3.6-27B never
treats as a jailbreak (0/21 refusals and recognitions over 10 think + 10 no-think samples + a
greedy comply), the share whose lens readouts contain a jailbreak recognition. Lower is better.

The items are WorkspaceBench's gated jailbreak negative controls (plan 0037), imported verbatim
with provenance, plus gated role-play candidates WSB did not take (``hb_only``). Judging runs
WSB's own pinned ``jailbreak_recognition`` family (jb-v3 recognition judge); HB ports none of
its code (``wsb.py`` is a pin check + subprocess calls). Not an index family: ``hbench index``
is unchanged. CLI ``hbench benign-roleplay import|run|report`` (``cli.register``)."""
