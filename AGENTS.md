# AGENTS.md

## Cursor Cloud specific instructions

### What this project is
CPU-only Python data-science challenge (Shipd / Project Eris): "FWS Refuge Evidence
Route Reconstruction", a RAG-style retrieval + ordering task. See `PROBLEM.md` for the
spec and `CLAUDE.md` for the hard prohibitions and required win-brief workflow.

- Deliverable: a single script run as `python3 solution.py <public_dir> <submission_out>`
  (paths from `sys.argv[1]`/`sys.argv[2]`). It reads the CSVs under `dataset/public/`
  (read-only) and writes `working/submission.csv` with columns
  `case_id,predicted_evidence_ids` (` | `-separated ordered evidence ids, capped at 6/case).
- There is no server/database/daemon — it's a batch script. Nothing to keep running.

### Environment / how deps are installed (non-obvious)
This is Ubuntu 24.04 with an externally-managed system Python 3.12 (PEP 668), and
`ensurepip`/`python3-venv` are NOT available, so a normal `python3 -m venv` fails.
The dev stack is therefore installed into the `ubuntu` user site with
`pip install --break-system-packages --user` (see the startup update script). `numpy`
is preinstalled system-wide; `pandas`, `scipy`, `scikit-learn`, `lightgbm` come from the
update script. Just use `python3` directly — the user-site packages are on its path.

### Run / smoke-test
`baseline_smoke_test.py` is an environment smoke test (NOT the graded solution). It loads
every dataset CSV, trains a small scikit-learn model on the training routes, and writes a
correctly formatted submission:

```
python3 baseline_smoke_test.py ./dataset/public ./working/submission.csv
```

Runs in a few seconds on CPU. A harmless `ConvergenceWarning` from LogisticRegression may
print; it does not affect the smoke test.

### Compliance caveats (important)
- The graded `solution.py`/`solution.ipynb` runs in the Kaggle Docker image and MUST NOT
  `pip install` anything — use only libraries bundled in that image. The local installs
  above are dev-only, chosen to mirror the Kaggle stack.
- Do not build the competitive solution before the win-brief workflow in `CLAUDE.md`
  (`prompts/02_CHALLENGE_PROMPT_BUILDER.md` → human GO). `baseline_smoke_test.py` is only a
  pipeline/format smoke test, not a leaderboard entry (a pure heuristic/lexical entry is
  disallowed as a submission — the graded artifact needs a trained ML model).
