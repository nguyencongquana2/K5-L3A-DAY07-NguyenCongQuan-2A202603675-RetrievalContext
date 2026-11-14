# CP3 dev error analysis

The selection rule was fixed before any test run: maximize Evidence Hit@3, then
Hit@1, then Context Evidence Rate, then minimize median retrieval latency.

## Retrieval changes

- **q06 improved:** TF-IDF top-3 missed the tuple-immutability evidence. Hybrid
  RRF brought page 44 into rank 3, and top-5 context therefore changed the case
  from no evidence to sufficient evidence. Dense semantic matching recovered the
  relation between "replaced by indexed assignment" and "immutable".
- **q09 improved:** TF-IDF top-3 retrieved module/script passages but missed the
  sentence about search-path order. Hybrid placed the page-52 gold chunk at rank
  2, changing Hit@3 and context evidence from false to true.
- **q10 regressed:** TF-IDF placed the `__all__` evidence at rank 3. RRF promoted
  other package/import chunks and pushed the gold chunk beyond top 5, despite the
  modules metadata filter. This case remains a selected-configuration failure.
- **q14 changed cutoff behavior:** the gold chunk moved from rank 2 to rank 4.
  Selected Hit@3 is false, but the top-5 context still contains the evidence.

## Agent/context errors

- The first CP3 run sent all five chunks to the 0.5B model. It ignored available
  evidence on q11, q13 and q15. Focusing the model on one chunk selected from the
  top-5 by question-term overlap improved q11, q12 and q15, and shortened answers.
- Focused selection still fails on ambiguous words. For q14 it selects the
  `for/else` passage instead of `try/else`; for q13 it selects a neighboring
  exception passage. q06 has the correct immutable-tuple chunk but the small model
  answers the polarity incorrectly. These are model/context-selection limits, not
  missing retrieval evidence.

No test query or label was inspected while making these changes.
