# CORRECT FRAMING — First law (most expensive mistake if skipped)

Framing is not mindset fluff. Framing is the question you answer **before** any model,
aug, ensemble, or CPU trick. Wrong framing makes every later hour compound interest on debt.

---

## The mistake

We asked:

> What model can we fine-tune quickly to produce a legal submission that moves val score?

Winners (implicitly) asked:

> What must a system compute, given the **definition of the label** and the **metric**
> and the **shift**, such that errors the metric punishes become structurally hard?

Same dataset. Different question. Different solution class.

---

## Correct framing procedure (do in order — no training)

Write answers in one screen. If you cannot, you are not ready to pick a backbone.

### F1. Label definition → required computation
- Quote how a label is defined (not how a tutorial solves “similar” tasks).
- Complete: **“To be correct, the model must represent ___.”**
- If deleting that blank makes the task undefined, that blank is non-negotiable structure.

### F2. Metric → error economics
- Quote the metric.
- Name which mistakes cost the most (rare tail, FP vs FN, order, calibration, …).
- Complete: **“A high score requires being good at ___, not merely at ___.”**

### F3. Shift → what must not be trusted
- Quote stated or implied train≠test shift.
- Complete: **“Any method that relies on ___ from train will fail on test.”**
- List mandatory defenses before any climb.

### F4. Supervision → what is allowed to assume
- Full instance labels vs weak bags vs pairs vs points.
- Complete: **“We are allowed to supervise ___; we are not allowed to assume ___.”**

### F5. Grader → what actually gets scored
- Retrain script vs CSV vs weights; device; budget.
- Complete: **“The thing we optimize in Colab is worthless unless it survives ___.”**
- Separately: list platform **hard bans** you will follow; if a rule is unclear for THIS
  challenge, **ask the human** (yes/no) — do not assume.

### F6. Forbidden first question
Never start with:
- “What Hub model is trending?”
- “What did we use last time on a vaguely related task?”
- “What gets a number on the board fastest?”

### F7. Leakage / group structure
- What would inflate holdout if ignored? (ids, row order, duplicates, patient/site/query groups, near-duplicate text/images, pretraining overlap)
- Complete: **“Split and features must respect ___; otherwise val lies.”**

### F8. Legal output constraints
- Exact submission schema: required fields, vocab, length, counts, `none`/empty rules, sorting, duplicates banned, etc.
- Complete: **“Every prediction must be decodable to ___; invalid means score ___.”**
- Empty/invalid policy belongs here (force-argmax, reject, default class) — not as an afterthought at submit.

### F9. Cost asymmetry (beyond F2)
- When FP and FN (or precision vs recall, or rare vs head) are not equal under the metric or prize.
- Complete: **“Prefer erring toward ___ because ___.”**
- If asymmetric, decision policy must say so; if symmetric, write SYMMETRIC.

### F10. Calibration / score reliability (optional — fill or N/A)
- Do raw scores need to be comparable across classes, folds, or time?
- Complete: **“Scores are used as ___ (rank-only | calibrated probs | margins); therefore ___.”**
- N/A if only hard labels are scored and ranking within-row is enough.

### F11. Multi-task / auxiliary objectives (optional — fill or N/A)
- Is there a useful auxiliary head (length, domain, presence-vs-type, retrieval then rerank)?
- Complete: **“Auxiliary ___ is allowed only if it serves F1/F2; otherwise omit.”**
- N/A if single-task is clearly sufficient.

### F12. External / retrieval / knowledge side-channel (optional — fill or N/A)
- Is retrieval, gazetteer, ontology, or metadata fusion load-bearing vs forbidden?
- Complete: **“Side channel ___ is {required | helpful-aux | forbidden} because ___.”**
- N/A if pure end-to-end on the given modality is the only honest path.

**Required every contest:** F1–F9.  
**Also keep on the card:** F10–F12 as filled answers **or** explicit `N/A — why`.

Start with F1–F12 (N/A allowed only on F10–F12). Only then: formulations.

---

## Framing test (kill criterion)

Your Primary formulation fails framing if:

1. It still works as a paragraph after you remove the label-structure sentence (F1) — then it ignores the task.
2. It does not mention the shift defense when a shift is stated (F3).
3. Its success is defined only as “val↑” with no link to metric error economics (F2/F9).
4. It is a tutorial stack with the contest output glued on at the end (fails F1/F8).
5. Its val protocol ignores grouping/leakage called out in F7.
6. It has no legal decode path for F8 constraints.

If any hold, **do not train**. Reframe.

---

## One-line doctrine

**Frame the computation the label and metric demand. Then choose models that implement that computation under the shift and the grader. Never the reverse.**
