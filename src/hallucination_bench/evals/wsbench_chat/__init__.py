"""wsbench_chat: WorkspaceBench's hallucination (chat) eval, ported, plus the taxonomy stage.

149 LMSYS / DailyDialog chat items answered by Qwen3.6-27B, 1,123 read sites. Per cell (arm x
item x site x layer): (summarise, token arms only) -> v5c span stage -> v5c verify stage ->
taxonomy-v1 classify stage. See ``plans/0001-bootstrap-wsbench-chat-taxonomy.md``.
"""

NAME = "wsbench_chat"
