"""Environment smoke-test baseline for FWS Refuge Evidence Route Reconstruction.

This is NOT the graded competition solution. It exists to prove the local
development environment works end to end: it loads every dataset CSV with
pandas, trains a small scikit-learn model on the provided training routes, and
writes a correctly formatted submission file.

The real graded `solution.py` must follow the win-brief workflow in CLAUDE.md.

Usage (mirrors the challenge runtime contract):
    python3 baseline_smoke_test.py <public_dir> <submission_out>

Local example:
    python3 baseline_smoke_test.py ./dataset/public ./working/submission.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def family_of(evidence_id: str) -> str:
    parts = evidence_id.split("_")
    return parts[1].upper() if len(parts) > 2 else "UNKNOWN"


def parse_family_counts(raw: str) -> dict:
    counts = {}
    if isinstance(raw, str):
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            fam, num = chunk.split(":")
            counts[fam.strip().upper()] = int(num.strip())
    return counts


def build_family_training_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, case in train.iterrows():
        counts = parse_family_counts(case.get("family_counts", ""))
        target_ids = str(case["target_evidence_ids"]).split(" | ")
        target_families = {family_of(t) for t in target_ids if t}
        for fam, fam_count in counts.items():
            rows.append(
                {
                    "case_type": case["case_type"],
                    "family": fam,
                    "family_count": fam_count,
                    "candidate_evidence_count": case["candidate_evidence_count"],
                    "region": str(case["region"]),
                    "label": int(fam in target_families),
                }
            )
    return pd.DataFrame(rows)


def train_family_scorer(train: pd.DataFrame) -> Pipeline:
    table = build_family_training_table(train)
    features = ["case_type", "family", "region", "family_count", "candidate_evidence_count"]
    cat = ["case_type", "family", "region"]
    num = ["family_count", "candidate_evidence_count"]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)]
    )
    model = Pipeline(
        [("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )
    model.fit(table[features], table["label"])
    acc = model.score(table[features], table["label"])
    print(f"[smoke] trained LogisticRegression family scorer on {len(table)} rows; train acc={acc:.3f}")
    return model


def order_hints(case_intent: pd.DataFrame) -> dict:
    hints = {}
    for _, row in case_intent.iterrows():
        fams = [f.strip().upper() for f in str(row["family_order_hint"]).split("|")]
        hints[row["case_type"]] = fams
    return hints


def predict(test, evidence_by_sanctuary, model, hints, max_ids):
    predictions = []
    for _, case in test.iterrows():
        pool = evidence_by_sanctuary.get(case["sanctuary_id"])
        if pool is None or pool.empty:
            predictions.append("")
            continue
        counts = parse_family_counts(case.get("family_counts", ""))
        fams_present = [f for f in counts if f in set(pool["source_family"].str.upper())]
        if not fams_present:
            fams_present = sorted(pool["source_family"].str.upper().unique())
        feat = pd.DataFrame(
            {
                "case_type": case["case_type"],
                "family": fams_present,
                "region": str(case["region"]),
                "family_count": [counts.get(f, 0) for f in fams_present],
                "candidate_evidence_count": case["candidate_evidence_count"],
            }
        )
        scores = model.predict_proba(feat)[:, 1]
        hint = hints.get(case["case_type"], [])
        hint_rank = {f: i for i, f in enumerate(hint)}
        order = sorted(
            range(len(fams_present)),
            key=lambda i: (-scores[i], hint_rank.get(fams_present[i], len(hint))),
        )
        ordered_families = [fams_present[i] for i in order]
        chosen = []
        upool = pool.assign(_fam=pool["source_family"].str.upper())
        for fam in ordered_families:
            fam_rows = upool[upool["_fam"] == fam]
            if not fam_rows.empty:
                chosen.append(fam_rows.iloc[0]["evidence_id"])
            if len(chosen) >= max_ids:
                break
        predictions.append(" | ".join(chosen[:max_ids]))
    return predictions


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    evidence = pd.read_csv(public_dir / "evidence_catalog.csv")
    case_intent = pd.read_csv(public_dir / "case_intent_catalog.csv")
    print(f"[smoke] loaded train={len(train)} test={len(test)} evidence={len(evidence)} rows")

    max_ids = int(case_intent["max_scored_evidence_ids"].max())
    hints = order_hints(case_intent)
    evidence_by_sanctuary = dict(tuple(evidence.groupby("sanctuary_id")))

    model = train_family_scorer(train)
    preds = predict(test, evidence_by_sanctuary, model, hints, max_ids)

    submission = pd.DataFrame({"case_id": test["case_id"], "predicted_evidence_ids": preds})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    non_empty = (submission["predicted_evidence_ids"].str.len() > 0).sum()
    print(f"[smoke] wrote {submission_out} rows={len(submission)} non_empty_predictions={non_empty}")


if __name__ == "__main__":
    main()
