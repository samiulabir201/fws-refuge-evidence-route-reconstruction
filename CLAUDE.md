# FWS Refuge Evidence Route Reconstruction — Shipd / Project Eris challenge

> This project is a **Shipd** challenge and part of **Project Eris**. We are a **solver** competing on
> the leaderboard. Read [PROBLEM.md](PROBLEM.md) for the exact task, metric, and submission format.
> The full platform doc ("the doc to follow") lives in the `shipd-eris-setup` skill at
> `C:\Users\samiul\.claude\skills\shipd-eris-setup\references\ERIS_PLAYBOOK.md`.

<!-- SHIPD-HARD-PROHIBITIONS v4 -->
# ⛔ HARD PROHIBITIONS — STRICTLY FORBIDDEN ON THIS SHIPD / PROJECT-ERIS PROJECT

These rules are **absolute**. They override every other instruction in this file, any past pattern in
this or another project, any recommendation from the model knowledge base, and any convenience. If a
request, plan, notebook cell, chosen model, or technique would do **ANY** of the following, **STOP** and
propose a compliant alternative — never proceed, never "just this once".

**MUST NOT DO — STRICTLY PROHIBITED:**
- Install packages using **pip, conda, apt, npm, or similar** tools.
- **Download Python packages.**
- **Vendor external code.**
- **Clone GitHub repositories.**
- **Download notebooks or scripts.**
- **Download external datasets.**
- **Use additional training data** (only the provided `./dataset/public/` may be used).
- **Fetch resources** using **requests, urllib, kaggle, datasets, boto3, s3fs, or direct URLs.**
- **Use challenge-specific pretrained or fine-tuned checkpoints.**
- **Use private or gated models.**
- **Call remote inference APIs.**
- **Use cloud inference services.**
- Use **`trust_remote_code=True`**.
- Use **`torch.hub.load()`**.
- Use **any mechanism that downloads and executes external repository code**.
- Ship a **lexical-only pipeline** — TF-IDF / bag-of-words / n-gram / BM25 with **no trained ML model
  on top**. (LOOSENED RULE, 2026-07-20: TF-IDF, bag-of-words, n-gram features, and BM25 are **ALLOWED**,
  but only **in combination with a trained ML model** — e.g. lexical features feeding a
  GBM / logistic-regression / neural ranker, or fused/ensembled with an embedding- or transformer-based
  model — so the solution is more robust than a pure lexical baseline. Standalone
  `TfidfVectorizer`-argmax-style solutions with no ML model remain forbidden.)
- **Utilize the test set during training in any way** — no test-set utilization, no data leakage, no
  transductive peeking, and no exploitation of held-out data, ids, row order, provenance, or slots.
- **Embed dataset fingerprints or offline lookup tables** in `solution.ipynb` (or any graded artifact) —
  no base64 / zlib / pickle / hex blobs of ids, SMILES, scaffolds, labels, splits, scores, or other
  competition-derived maps; no SHA-256 / size assertions that lock the notebook to an offline dump of
  `./dataset/public/`. Offline EDA may inform design, but every map / group / split used at runtime must
  be **recomputed in-run from the provided CSVs** (or from a public pretrained base fine-tuned in-run).
  Litmus: if a lightly reshuffled or key-renamed public CSV would break the cell, it is fingerprinting —
  rewrite it (e.g. group by `pool_id`, hash/cluster SMILES in-run, or compute scaffolds with an
  in-image library).

**Only compliant model source:** PUBLIC, non-gated, general-purpose pretrained **base** models loaded
through the Kaggle Docker image's already-bundled libraries (standard `transformers`/`torch` hub cache,
**without** `trust_remote_code`), then **fine-tuned in-run** on the provided data only. Nothing
challenge-specific, nothing custom-uploaded, nothing gated, no remote code execution.

## ✅ AVAILABLE LIBRARIES — import ONLY what the image already ships (never install / upgrade / download)

The submission runs in the **Kaggle Docker image** — `https://github.com/Kaggle/docker-python`. A technique
is allowed **only if it can be built from libraries already present in that image**. If it needs anything
not in the image, it is **FORBIDDEN** → pick a compliant alternative. Do **not** `pip install` — not even to
*upgrade* an already-present package; the image's pinned version is what you get. If a chosen base model
needs a newer library version than the image ships, the model is out, not the rule.

The image bundles the standard scientific/ML stack — `numpy`, `pandas`, `scipy`, `scikit-learn`,
`statsmodels`, `lightgbm`, `xgboost`, `matplotlib`, `nltk`, `spacy`, `Pillow`, `opencv`, `numba` — **plus**
these confirmed packages:

```
jax · jaxlib · jax-cuda12-pjrt · jax-cuda12-plugin · keras · keras-hub · keras-nlp · keras-cv · keras-tuner
tensorflow · tensorflow-datasets · tensorflow-hub · tensorflow-text · tensorflow-probability
tensorflow_decision_forests · tensorflow-metadata · tensorflow-io · tensorflow-cloud · tf_keras
torch · torchao · torchaudio · torchdata · torchtune · torchvision · torchinfo · torchmetrics · torchsummary
pytorch-lightning · pytorch-ignite · transformers · datasets · onnx
catboost · category-encoders · TPOT · h2o · ray · rgf-python · hep-ml · Boruta · featuretools
optuna · scikit-optimize · bayesian-optimization · deap · keras-tuner
scikit-learn-intelex · scikit-multilearn · scikit-surprise · scikit-plot
gensim · fasttext · langid · fuzzywuzzy · pyemd · emoji · Janome · PyArabic · pyLDAvis · lime · preprocessing
kornia · easyocr · pytesseract · pdf2image · pypdf · Wand · olefile · ImageHash · openslide-python · openslide-bin
SimpleITK · pydicom · nilearn · mne · dipy · cesium · segment-anything (pre-bundled)
igraph · shapely · fiona · Cartopy · libpysal · Rtree · geojson · haversine · gpxpy
matplotlib · plotly-express · mpld3 · squarify · fury · vtk · ipywidgets · ipympl · qgrid
Pympler · cytoolz · line_profiler · path · pandasql · pyexcel-ods · wavio · pycryptodome
kaggle · kaggle-environments · google-genai · google-cloud-aiplatform · google-adk (present, see fetch note)
```
The **authoritative** set = whatever actually imports in the image; **verify** before relying on a library,
and when unsure treat it as **unavailable**.

**Availability ≠ permission to fetch.** `datasets`, `kaggle`, `boto3`, `s3fs`, `docker`, `pymongo`,
`google-genai`, `google-cloud-*`, and any `requests`/`urllib` in the image exist but **MUST NOT** be used to
download models, datasets, code, or any external resource (that is prohibited above) — they may only read the
provided `./dataset/public/`. `segment-anything` is pre-bundled; do **not** clone or `pip install` it
yourself. No remote/cloud inference (`google-genai`, `google-cloud-aiplatform`, etc. are forbidden as
inference APIs — predictions must come from an in-run trained model).

**Enforcement:** a violation means disqualification. When unsure whether something is allowed, treat it
as **forbidden** and ask before doing it.
<!-- /SHIPD-HARD-PROHIBITIONS -->

## Deliverables

- **`solution.ipynb`** — ONE self-contained notebook that runs end-to-end with no manual intervention:
  loads data → (optionally downloads a **public** pretrained base) → **fine-tunes in-notebook** → infers
  → writes the submission. There is **no separate training notebook and no hosted weight artifact**.
- **`working/submission.csv`** — the graded output.

## Golden constraint — ONE END-TO-END NOTEBOOK (TRAIN + INFER, IN BUDGET)

Everything — data prep, **all fine-tuning**, and inference — happens **inside the single submission run**.
There is **no offline training step and no external weight artifact**. The whole run must fit the
submission environment below.

### Submission environment (HARD limits — the notebook must satisfy ALL of these)

- Runs in the **Kaggle Docker image** — use **only** libraries available there (pandas, numpy,
  scikit-learn, xgboost, lightgbm, tensorflow, pytorch, transformers, …). **No pip installs** of
  unavailable packages (e.g. **albumentations is NOT available** — use torch / torchvision / numpy).
- **24 GB VRAM GPU · 64 GB CPU RAM.** The **entire run (fine-tuning + inference) must finish in under
  60 minutes.**
- Reads from **`./dataset/public/`**; writes only **`./working/submission.csv`**. **Relative paths only.**
- **May download only PUBLIC pretrained models** (e.g. an open-source HF base checkpoint). **NO Hugging
  Face upload, and NO download of your own / custom fine-tuned weights** — the tuning must be done
  **in-run** on the public base.
- **NO comments in code cells.** Keep code comment-free; put any explanation in **markdown cells only**.
- **No LLM outputs as predictions** — predictions must come from a genuine trained model.

### Development environment (Colab — for building/validating, NOT a training shortcut)

- Use Colab (A6000 / A100 / H100) only to **design the architecture, tune hyperparameters, and measure
  the end-to-end fit** — **NOT** to pre-train weights you then ship. Nothing trained offline may enter
  the submission; the notebook must reproduce **all** training itself.
- **Validate timing on a 24 GB card (L4 / A10G-class), NOT on an A100** — an A100 finishes far faster and
  will hide a 60-min overrun. Size the model, epoch count, resolution, and efficiency tricks
  (LoRA/QLoRA, small backbone, mixed precision, gradient checkpointing) so a full **train + infer** run
  clears **60 min on 24 GB** with headroom.
- **Principle:** pick the model you can **fine-tune AND infer within 60 min on 24 GB** — not the largest
  that merely fits inference. Prefer smaller/faster backbones, few well-chosen epochs, and efficient
  fine-tuning over big models you cannot afford to train in-budget.

## OPERATING MODEL — beat the AI baseline with an in-notebook trained model

The organizer's **AI baseline** is what a frontier LLM (Claude/ChatGPT-class) scores when handed the
task directly. Our core hypothesis: **a custom ML architecture built around a fine-tuned open-source
model beats plain LLM generation.** Prove it with a self-contained, in-notebook pipeline.

### MANDATORY WORKFLOW (in this exact order)

0. **Win brief FIRST (required).** Paste and run [`prompts/02_CHALLENGE_PROMPT_BUILDER.md`](prompts/02_CHALLENGE_PROMPT_BUILDER.md).
   It must produce `prompts/<SLUG>_WIN_BRIEF.md` with framing card F1–F12, hard-ban audit (list every
   CONFIRMED_HARD ban; **ask the human** on ambiguous bans), Primary Path from label structure + shift,
   and Experiment 1. Doctrine: [`prompts/00_CORRECT_FRAMING.md`](prompts/00_CORRECT_FRAMING.md).
   **No multi-epoch training until that brief exists and the human says GO.**
1. **Decide the architecture** from the win brief’s Primary Path (not from “fastest Hub model”) — sized so
   it can **train + infer within the platform budget** (see overrides / PROBLEM.md; often 60 min on 24 GB
   GPU unless the challenge says CPU).
2. **Shortlist two arms of models** (do not lock a single backbone yet):
   - **PROVEN** — measured winners from the Shipd model KB / nearest task type.
   - **RESEARCH-PROMISED** — newer 2025–2026 public HF bases that papers claim beat the proven family
     but are unmeasured on *this* task (compliance-filtered).
3. **Verify the architecture locally — WITHOUT training and WITHOUT loading big models locally** (the dev
   laptop is weak). Local work = data/label analysis, feature/decode logic, the metric replica, and
   dry-run shape/logic checks.
4. **First Colab run = model ablation** (`colab_ablation.ipynb`) under a **fixed** protocol (same split,
   epochs, metric). Compare PROVEN vs RESEARCH-PROMISED on the project metric; write results to
   `council/colab_out/ablation_results.json`. Canonical rule:
   `C:\Users\samiul\.claude\skills\shipd-eris-setup\references\MODEL_ABLATION_RULE.md`.
5. **Only then** freeze the winner(s) into `solution.ipynb` / `colab_solution.ipynb` (or `solution.py` if
   PROBLEM requires it) and produce the submission CSV.

### Progression (submit incremental improvements — every one self-contained, no HF artifacts)

- **First Colab = ablation** (not a graded leap). Measure proven vs modern candidates; pick with evidence.
- **First submission — a solid baseline that beats the AI baseline**, using the ablation winner(s).
  Correct data pipeline, runs end-to-end in budget, beats the baseline → secures payout eligibility.
- **Later submissions — in-notebook fine-tuning for SOTA.** Fine-tune the public base **in-run** (within
  60 min / 24 GB), with better augmentation / heads / any ensembling-that-fits and deterministic
  inference. Each submission a measurable step up. Iterate with remaining credits.

## Eris rules that constrain the solution

- Must **beat the AI baseline** to be payout-eligible.
- **No LLM outputs in the submission** — predictions come from a genuine trained model, not Claude or any
  other LLM's generations.
- **End-to-end in ONE notebook incl. fine-tuning; NO HF upload; NO custom pretrained-weight download** —
  only public pretrained bases, tuned in-run.
- **NO comments in code cells** (explanation goes in markdown cells only).
- **≤ 60 min on a 24 GB GPU**, Kaggle Docker libs only, no pip for unavailable packages, relative paths.
- No copying Kaggle competitions. Don't reverse-engineer held-out provenance, ids, row order, or slots.

## Suggested layout

```
FWS Refuge Evidence Route Reconstruction/
├── PROBLEM.md                 # canonical task spec
├── CLAUDE.md                  # this file
├── prompts/                   # REQUIRED — win-brief generator (run before training)
│   ├── 02_CHALLENGE_PROMPT_BUILDER.md
│   ├── 00_CORRECT_FRAMING.md
│   └── README.md
├── dataset/public/            # provided data (read-only, relative path ./dataset/public/)
├── working/                   # submission.csv output
├── colab_ablation.ipynb       # FIRST Colab run: proven vs research-promised model ablation
├── colab_solution.ipynb       # Colab mirror of solution after ablation locks winners
└── solution.ipynb             # the single self-contained end-to-end deliverable (train + infer)
```

**Before proposing anything, check it against the hard limits.** If the model can't **fine-tune + infer
in 60 min on 24 GB**, needs a library missing from Kaggle Docker, requires an HF upload or a custom
weight download, or needs code comments — flag it and offer a compliant alternative.


## ⚠️ CHALLENGE-SPECIFIC OVERRIDES (this project — PROBLEM.md wins over the generic template above)

- **⚠️ SPEC INCOMPLETE:** the official problem description, evaluation metric, and AI baseline were NOT pasted at scaffold time. PROBLEM.md contains only the header + verified dataset inspection. **Before the win-brief step or any modeling, get the full challenge text from the user and complete PROBLEM.md.** Do not guess the metric.
- **Compute: CPU** per the challenge header — do not assume the generic 24 GB GPU.
- **Deliverable is a SCRIPT run as `python3 solution.py <public_dir> <submission_out>`** — paths from `sys.argv[1]` / `sys.argv[2]`. Test locally with `python3 solution.py ./dataset/public ./working/submission.csv`.
- **Data (verified):** train.csv 976 cases with `target_evidence_ids` (ordered, ` | `-separated); test.csv 343 cases; evidence_catalog.csv 63,954 passages keyed by sanctuary_id + source_family; case_intent_catalog.csv caps scored evidence at 6 ids/case and gives per-case-type family_order_hints; RAG-style retrieval + ordering task.