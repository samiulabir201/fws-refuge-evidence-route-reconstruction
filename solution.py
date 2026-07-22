import hashlib
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy import sparse

BUCKET = {
    "fractional": 0,
    "tiny": 1,
    "small": 2,
    "medium": 3,
    "large": 4,
    "very_large": 5,
    "huge": 6,
}

KWS = [
    "center",
    "entrance",
    "office",
    "trailhead",
    "main",
    "visitor",
    "parking",
    "pkg",
    "restroom",
    "contact",
    "station",
    "auto",
    "tour",
    "hatchery",
    "gate",
    "road",
    "trail",
    "water",
    "canal",
    "ditch",
    "berm",
    "culvert",
    "bridge",
    "shop",
    "hq",
    "headquarters",
    "ada",
    "public",
    "condition",
    "concrete",
    "gravel",
    "excellent",
    "good",
    "minor",
    "loop",
    "system",
    "refuge",
    "numeric",
    "field",
    "marsh",
    "pond",
    "lake",
    "blind",
    "fence",
    "property",
    "building",
    "access",
    "toilet",
    "hunter",
    "hunting",
    "viewing",
    "photo",
    "cattle",
    "seasonal",
    "paved",
    "lollipop",
    "boardwalk",
    "levee",
    "control",
    "inlet",
    "outlet",
]

CATS = [
    "surface type",
    "user access",
    "access type",
    "gate type",
    "route type",
    "condition",
    "structure type",
    "fence type",
    "blind use",
    "property type",
    "managed use",
    "class",
    "functional class",
    "route name",
    "building name",
    "property name",
    "gate name",
    "material",
    "structure type 2",
    "in/out",
    "suitability",
    "activities",
    "estimated time",
    "construction",
    "unpaved",
    "public use",
]

REL_BASE = ["len", "n_fields", "nb_mean", "nb_max", "kw_count_sum", "tfidf_cos", "struct_cos", "token_prior"]
UNIQ_KWS = [
    "center",
    "entrance",
    "office",
    "trailhead",
    "contact",
    "station",
    "auto",
    "tour",
    "pkg",
    "parking",
    "visitor",
    "hatchery",
]


def parse_fields(text):
    d = {}
    for m in re.finditer(r"([A-Za-z0-9 /_\'()-]+):\s*([^.]*)\.", text):
        d[m.group(1).strip().lower()] = m.group(2).strip()
    return d


def field_tokens(d):
    toks = []
    for k, v in d.items():
        k2 = k.replace(" ", "_")
        if len(v) < 48 and "=" not in v:
            toks.append(f"{k2}={v}")
        if "numeric_buckets=" in v:
            m = re.search(r"numeric_buckets=([a-z_]+)", v)
            if m:
                toks.append(f"{k2}_nb={m.group(1)}")
        if "keywords=" in v:
            m = re.search(r"keywords=([^;]+)", v)
            if m:
                for w in m.group(1).strip().split():
                    toks.append(f"{k2}_kw={w}")
        if "attribute_signal=" in v:
            m = re.search(r"attribute_signal=([a-z_]+)", v)
            if m:
                toks.append(f"{k2}_sig={m.group(1)}")
        if "keyword_count=" in v:
            m = re.search(r"keyword_count=(\d+)", v)
            if m:
                toks.append(f"{k2}_kc={m.group(1)}")
    return toks


def cat_hash(key, val):
    return float(int(hashlib.md5(f"{key}={val}".encode()).hexdigest()[:8], 16) % 256)


def dense_features(passage, case_type, family, token_prior_map):
    d = parse_fields(passage)
    text = passage.lower()
    f = {}
    for kw in KWS:
        f[f"kw_{kw}"] = float(kw in text)
    f["len"] = float(len(passage))
    f["n_fields"] = float(len(d))
    f["has_sig_present"] = float("attribute_signal=present" in text)
    f["has_priority"] = float("priority_signal" in text and "present" in text)
    f["has_handicap"] = float("handicap" in text and "present" in text)
    pu = d.get("public use", "").lower()
    f["public_use"] = float(("use" in pu) and ("non" not in pu))
    f["unpaved"] = float(d.get("unpaved", "").lower() == "yes")
    nbs = re.findall(r"numeric_buckets=([a-z_]+)", text)
    f["nb_mean"] = float(np.mean([BUCKET.get(x, 3) for x in nbs])) if nbs else 0.0
    f["nb_max"] = float(max([BUCKET.get(x, 3) for x in nbs])) if nbs else 0.0
    f["kw_count_sum"] = float(sum(int(x) for x in re.findall(r"keyword_count=(\d+)", text)))
    for key in CATS:
        f[f"cat_{key.replace(' ', '_')}"] = cat_hash(key, d.get(key, ""))
    toks = field_tokens(d)
    prior = 0.0
    for t in toks:
        prior += token_prior_map.get((case_type, family, t), token_prior_map.get(("*", family, t), 0.0))
    f["token_prior"] = prior
    f["n_tokens"] = float(len(toks))
    return f, toks


def build_ranking_frame(cases, ev_by_sanct, intent_sum, token_prior_map, hint=None, labeled=True):
    rows = []
    gid = 0
    for _, row in cases.iterrows():
        cand = ev_by_sanct[row.sanctuary_id]
        if labeled:
            tids = set(x.strip() for x in str(row.target_evidence_ids).split("|"))
            id2fam = dict(zip(cand.evidence_id, cand.source_family))
            fams = [f for f in hint[row.case_type] if f in set(id2fam[t] for t in tids)]
        else:
            available = set(cand.source_family.unique())
            fams = [f for f in hint[row.case_type] if f in available]
            tids = set()
        for fam in fams:
            sub = cand[cand.source_family == fam]
            if len(sub) == 0:
                continue
            for _, e in sub.iterrows():
                feats, toks = dense_features(e.passage, row.case_type, fam, token_prior_map)
                feats["label"] = int(e.evidence_id in tids) if labeled else 0
                feats["case_type"] = row.case_type
                feats["family"] = fam
                feats["evidence_id"] = e.evidence_id
                feats["case_id"] = row.case_id
                feats["group"] = gid
                feats["passage"] = e.passage
                feats["struct_text"] = " ".join(toks)
                feats["query"] = f"{row.case_type} {intent_sum[row.case_type]} family={fam}"
                rows.append(feats)
            gid += 1
    return pd.DataFrame(rows)


def add_relative_features(df):
    parts = []
    for _, gg in df.groupby("group", sort=False):
        g = gg.copy()
        for c in REL_BASE:
            if c not in g.columns:
                continue
            g[c + "_rank"] = g[c].rank(ascending=False) / len(g)
            g[c + "_z"] = (g[c] - g[c].mean()) / (g[c].std() + 1e-6)
        g["group_size"] = float(len(g))
        for kw in UNIQ_KWS:
            col = f"kw_{kw}"
            g[col + "_uniq"] = g[col] / (g[col].sum() + 1e-6)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def group_sizes(sub):
    return sub.groupby("group", sort=False).size().tolist()


def learn_token_priors(train, ev_by_sanct):
    pos = {}
    neg = {}
    for _, row in train.iterrows():
        tids = set(x.strip() for x in str(row.target_evidence_ids).split("|"))
        cand = ev_by_sanct[row.sanctuary_id]
        id2fam = dict(zip(cand.evidence_id, cand.source_family))
        target_fams = {id2fam[t] for t in tids}
        for fam in target_fams:
            sub = cand[cand.source_family == fam]
            if len(sub) <= 1:
                continue
            for _, e in sub.iterrows():
                toks = field_tokens(parse_fields(e.passage))
                bucket = pos if e.evidence_id in tids else neg
                for t in set(toks):
                    key = (row.case_type, fam, t)
                    bucket[key] = bucket.get(key, 0) + 1
                    key2 = ("*", fam, t)
                    bucket[key2] = bucket.get(key2, 0) + 1
    priors = {}
    keys = set(pos) | set(neg)
    for k in keys:
        s = pos.get(k, 0)
        n = neg.get(k, 0)
        priors[k] = (s + 1.0) / (s + n + 2.0) - 0.5
    return priors


def predict_routes(cases, df_scores, hint, max_n=6):
    out = []
    score_map = {}
    for _, r in df_scores.iterrows():
        score_map[(r.case_id, r.family, r.evidence_id)] = float(r.score)
    fam_ids = {}
    for _, r in df_scores.iterrows():
        fam_ids.setdefault((r.case_id, r.family), []).append(r.evidence_id)
    for _, row in cases.iterrows():
        picked = []
        for fam in hint[row.case_type]:
            ids = fam_ids.get((row.case_id, fam), [])
            if not ids:
                continue
            best = max(ids, key=lambda eid: score_map[(row.case_id, fam, eid)])
            picked.append(best)
            if len(picked) >= max_n:
                break
        out.append(
            {
                "case_id": row.case_id,
                "predicted_evidence_ids": " | ".join(picked),
            }
        )
    return pd.DataFrame(out)


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    evidence = pd.read_csv(public_dir / "evidence_catalog.csv")
    intent = pd.read_csv(public_dir / "case_intent_catalog.csv")

    hint = {
        r.case_type: [x.strip() for x in str(r.family_order_hint).split("|")]
        for _, r in intent.iterrows()
    }
    intent_sum = {r.case_type: r.intent_summary for _, r in intent.iterrows()}
    max_n = int(intent["max_scored_evidence_ids"].iloc[0])
    ev_by_sanct = {sid: g.reset_index(drop=True) for sid, g in evidence.groupby("sanctuary_id")}

    token_prior_map = learn_token_priors(train, ev_by_sanct)

    train_df = build_ranking_frame(
        train, ev_by_sanct, intent_sum, token_prior_map, hint=hint, labeled=True
    )

    case_types = sorted(train_df.case_type.unique())
    families = sorted(train_df.family.unique())
    ct_map = {c: i for i, c in enumerate(case_types)}
    fam_map = {c: i for i, c in enumerate(families)}
    train_df["case_type_i"] = train_df.case_type.map(ct_map).astype(float)
    train_df["family_i"] = train_df.family.map(fam_map).astype(float)

    tf_pass = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    Xp = tf_pass.fit_transform(train_df["passage"])
    Xq = TfidfVectorizer(vocabulary=tf_pass.vocabulary_).fit_transform(train_df["query"])
    train_df["tfidf_cos"] = np.asarray(normalize(Xp).multiply(normalize(Xq)).sum(axis=1)).ravel()

    tf_struct = TfidfVectorizer(max_features=12000, token_pattern=r"[^ ]+", min_df=2)
    Xs = tf_struct.fit_transform(train_df["struct_text"])
    Xqs = TfidfVectorizer(vocabulary=tf_struct.vocabulary_).fit_transform(
        train_df["query"] + " " + train_df["family"]
    )
    train_df["struct_cos"] = np.asarray(normalize(Xs).multiply(normalize(Xqs)).sum(axis=1)).ravel()

    train_df = add_relative_features(train_df)

    meta = {
        "label",
        "case_type",
        "family",
        "evidence_id",
        "case_id",
        "group",
        "passage",
        "query",
        "struct_text",
    }
    feat_cols = [c for c in train_df.columns if c not in meta]

    X_lr = sparse.hstack([Xs, Xp]).tocsr()
    lr = LogisticRegression(max_iter=200, C=1.0, solver="saga")
    lr.fit(X_lr, train_df["label"].values)

    dtrain = lgb.Dataset(
        train_df[feat_cols],
        label=train_df["label"].values,
        group=group_sizes(train_df),
    )
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1],
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=10,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbosity=-1,
        lambdarank_truncation_level=20,
    )
    model = lgb.train(params, dtrain, num_boost_round=280)

    test_df = build_ranking_frame(
        test, ev_by_sanct, intent_sum, token_prior_map, hint=hint, labeled=False
    )

    test_df["case_type_i"] = test_df.case_type.map(ct_map).fillna(-1).astype(float)
    test_df["family_i"] = test_df.family.map(fam_map).fillna(-1).astype(float)

    Xp_te = TfidfVectorizer(vocabulary=tf_pass.vocabulary_).fit_transform(test_df["passage"])
    Xq_te = TfidfVectorizer(vocabulary=tf_pass.vocabulary_).fit_transform(test_df["query"])
    test_df["tfidf_cos"] = np.asarray(normalize(Xp_te).multiply(normalize(Xq_te)).sum(axis=1)).ravel()

    Xs_te = TfidfVectorizer(vocabulary=tf_struct.vocabulary_).fit_transform(test_df["struct_text"])
    Xqs_te = TfidfVectorizer(vocabulary=tf_struct.vocabulary_).fit_transform(
        test_df["query"] + " " + test_df["family"]
    )
    test_df["struct_cos"] = np.asarray(normalize(Xs_te).multiply(normalize(Xqs_te)).sum(axis=1)).ravel()

    test_df = add_relative_features(test_df)
    X_lr_te = sparse.hstack([Xs_te, Xp_te]).tocsr()
    lr_score = lr.predict_proba(X_lr_te)[:, 1]

    for c in feat_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0
    lgb_pred = model.predict(test_df[feat_cols])

    parts = []
    tmp = test_df.copy()
    tmp["lgb_raw"] = lgb_pred
    tmp["lr_score"] = lr_score
    for _, gg in tmp.groupby("group", sort=False):
        g = gg.copy()
        p1 = (g["lgb_raw"] - g["lgb_raw"].mean()) / (g["lgb_raw"].std() + 1e-6)
        p2 = (g["lr_score"] - g["lr_score"].mean()) / (g["lr_score"].std() + 1e-6)
        g["score"] = 0.75 * p1 + 0.25 * p2
        parts.append(g)
    test_df = pd.concat(parts, ignore_index=True)

    submission = predict_routes(test, test_df, hint, max_n=max_n)
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    submission = sample[["case_id"]].merge(submission, on="case_id", how="left")
    submission["predicted_evidence_ids"] = submission["predicted_evidence_ids"].fillna("")

    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)


if __name__ == "__main__":
    main()
