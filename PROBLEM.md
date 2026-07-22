# FWS Refuge Evidence Route Reconstruction

**Type:** Flag Problem
**Domain:** RAG
**Difficulty:** Medium
**Scoring:** ↑ Higher is better
**Compute:** CPU
**AI Baseline:** ⚠️ NOT PROVIDED YET

> ## ⚠️ INCOMPLETE SPEC — PASTE THE FULL CHALLENGE DESCRIPTION
>
> Only the challenge header and generic submission instructions were provided when this project was
> scaffolded. The official **Overview / Problem Description, Dataset section, Evaluation metric +
> formula, Submission format rules, What-Not-To-Use rules, and AI Baseline** are still missing.
> **Paste the full challenge text so this file can be completed BEFORE running the win-brief
> generator or building any solution.** Everything below the "Observed dataset structure" heading
> was derived by direct inspection of the extracted files and is verified, but it is NOT a
> substitute for the official spec (especially the metric definition and edge-case rules).

## Observed dataset structure (verified by inspection — not the official spec)

Files in `dataset/public/`:

| File | Shape | Columns |
| --- | --- | --- |
| train.csv | 976 × 10 | case_id, sanctuary_id, case_type, region, state, resource_type, candidate_evidence_count, family_counts, query_text, **target_evidence_ids** |
| test.csv | 343 × 9 | same as train minus target_evidence_ids |
| sample_submission.csv | 343 × 2 | case_id, predicted_evidence_ids (empty/NaN placeholder) |
| evidence_catalog.csv | 63,954 × 7 | evidence_id, sanctuary_id, source_family, region, state, resource_type, passage |
| case_intent_catalog.csv | 3 × 4 | case_type, intent_summary, family_order_hint, **max_scored_evidence_ids** |
| source_family_catalog.csv | 12 × 2 | source_family, description |

Observed facts:

- Each case asks to "Reconstruct an ordered refuge evidence route for intent …" (see `query_text`).
- `target_evidence_ids` in train.csv is a **pipe-separated (` | `) ordered list of evidence_ids** drawn from evidence_catalog.csv.
- Evidence ids are scoped by sanctuary: each case's `sanctuary_id` links to a pool of candidate evidence rows in evidence_catalog.csv (`candidate_evidence_count` gives the pool size; `family_counts` breaks it down by source_family, e.g. `ACCESS:3; BLIND:3; GATE:3; PARKING:16; …`).
- Three case types (case_intent_catalog.csv): `hydrology_repair_route`, `visitor_access_route`, `operations_facility_route` — each with an `intent_summary`, a `family_order_hint` (e.g. `WATER_LINE | WATER_POINT | ACCESS | GATE | ROAD | PARKING…`), and `max_scored_evidence_ids = 6`.
- 12 source families (ACCESS, BLIND, BUILDING, GATE, PARKING, PROPERTY, ROAD, TRAIL, TRAIL_NOTE, WATER_LINE, WATER_POINT, …) described in source_family_catalog.csv.
- Evidence `passage` is structured text, e.g. `Evidence family: PARKING. route number: numeric_buckets=…`.
- Submission: `case_id, predicted_evidence_ids` — presumably the same pipe-separated ordered format as `target_evidence_ids` (confirm from official spec).

## Expected Output

Your script receives the public dataset directory and exact submission CSV path as two positional arguments.

### Solution submission instructions

Upload one Python script or notebook and one sample submission CSV. You do not enter paths in the UI—the platform supplies them when your code runs.

**Runtime command**

```
python3 solution.py <public_dir> <submission_out>
```

**Starter pattern**

```python
import sys
from pathlib import Path
import pandas as pd

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")

# Train and predict...
submission_out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_out, index=False)
```

**Test locally**

```
python3 mycode.py ./dataset/public ./working/submission.csv
```

---

Dataset extracted under [dataset/public/](dataset/public/). Submission output → working/submission.csv.
