# Judge and caveats

- Every judge call defaults to `claude-opus-5-5` (overrides: `judge_model=`,
  `summarizer_model=`, `classifier_model=`, `map_model=`). The WorkspaceBench pin is Gemini 3.8
  Flash, so absolute span-stage (`v5c-chat`) rates are **not comparable** to published
  WorkspaceBench numbers.
- The Anthropic route sends no `temperature` (Opus 5.5 rejects it), so every label is
  a **single sample at the default temperature**. Opus 5.5 cannot disable thinking; calls omit
  `thinking` and send `output_config.effort = low` (`effort=`).
- Token lenses are judged through the summariser, which is told to "commit to the most specific
  reading": their labels describe the summary, and their unsupported / strict / lenient rates
  are reported in a separate table and not ranked against prose lenses.
- `sae-l42` feature descriptions and `template-v3` words are not byte-level BPE, so
  `decode_token` garbles any non-ASCII in them, and the summariser prompt calls them "token
  readouts"; this matches the WorkspaceBench source.
- `concept` for unsupported spans (invented events and stances) extends the issue's
  "entirely contradictory claims" wording.
- Calibration gold labels are written by Claude, the classifier's model family, so agreement
  is optimistic.
