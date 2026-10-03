"""
STEP 6 : Feature selection.

1. Rank features with Random Forest importance (and Mutual Information as a second opinion).
   Ranking uses ONLY a part of the training data.
2. Try the top-k features for several k and measure F1 on a VALIDATION split (taken from training data).
3. Choose the smallest k whose F1 is within F1_TOLERANCE of the all-features F1.
The unseen test set is never touched here, so the final before/after comparison is fair.
"""
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

import config
from evaluate import compute_metrics


def _rf(n):
    return RandomForestClassifier(n_estimators=n, n_jobs=-1, class_weight="balanced_subsample",
                                  random_state=config.RANDOM_STATE)


def run_feature_selection(data):
    print("=" * 70)
    print("STEP 6: FEATURE SELECTION")
    print("=" * 70)
    X, y = data["X_train"], data["y_train"]
    X_sub, X_val, y_sub, y_val = train_test_split(
        X, y, test_size=config.VAL_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    # ---- ranking ----
    rf = _rf(100).fit(X_sub, y_sub)
    imp = rf.feature_importances_
    n_mi = min(30_000, len(X_sub))
    mi_idx = np.random.RandomState(config.RANDOM_STATE).choice(len(X_sub), n_mi, replace=False)
    mi = mutual_info_classif(X_sub.iloc[mi_idx], y_sub[mi_idx], random_state=config.RANDOM_STATE)
    rank = pd.DataFrame(dict(Feature=X.columns, RF_Importance=imp, MI_Score=mi))
    rank["RF_Rank"] = rank["RF_Importance"].rank(ascending=False, method="first").astype(int)
    rank["MI_Rank"] = rank["MI_Score"].rank(ascending=False, method="first").astype(int)
    rank = rank.sort_values("RF_Rank").reset_index(drop=True)
    rank["Cumulative_Importance"] = rank["RF_Importance"].cumsum()
    rank.round(6).to_csv(config.TABLES_DIR / "feature_ranking.csv", index=False)
    ordered = rank["Feature"].tolist()
    print("Top 10 features (Random Forest importance):")
    print(rank.head(10)[["RF_Rank", "Feature", "RF_Importance", "MI_Rank"]].round(4).to_string(index=False))

    # ---- sweep over k (validation set) ----
    n_feat = len(ordered)
    ks = sorted({k for k in config.K_VALUES if k < n_feat} | {n_feat})
    rows = []
    for k in ks:
        cols = ordered[:k]
        t0 = time.perf_counter()
        m = _rf(50).fit(X_sub[cols], y_sub)
        tt = time.perf_counter() - t0
        met = compute_metrics(y_val, m.predict(X_val[cols]))
        met.update(K=k, Train_Time_s=tt)
        rows.append(met)
        print(f"  k={k:>3}  F1={met['F1']:.4f}  Acc={met['Accuracy']:.4f}  train={tt:.1f}s")
    sweep = pd.DataFrame(rows)[["K", "Accuracy", "Precision", "Recall", "F1", "Train_Time_s"]]
    sweep.round(5).to_csv(config.TABLES_DIR / "k_sweep_validation.csv", index=False)

    full_f1 = sweep.loc[sweep["K"] == n_feat, "F1"].iloc[0]
    if config.FORCE_K:
        k_sel = min(int(config.FORCE_K), n_feat)
    else:
        ok = sweep[sweep["F1"] >= full_f1 - config.F1_TOLERANCE]
        k_sel = int(ok["K"].min())
    selected = ordered[:k_sel]
    with open(config.TABLES_DIR / "selected_features.json", "w") as f:
        json.dump(dict(k=k_sel, total_features=n_feat, features=selected), f, indent=2)
    print(f"\nSelected {k_sel} of {n_feat} features: {selected}")

    # ---- figures ----
    top = rank.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#d64545" if f in selected else "#9aa5b1" for f in top["Feature"]]
    ax.barh(top["Feature"], top["RF_Importance"], color=colors)
    ax.set_xlabel("Random Forest importance"); ax.set_title("Top 20 features (red = selected)")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "02_feature_importance.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["K"], sweep["F1"], "o-", color="#3b6fb6", label="F1 (validation)")
    ax.axhline(full_f1, ls="--", color="gray", label="All features")
    ax.axvline(k_sel, ls=":", color="#d64545", label=f"Selected k={k_sel}")
    ax.set_xlabel("Number of features (top-k)"); ax.set_ylabel("F1-score")
    ax2 = ax.twinx(); ax2.plot(sweep["K"], sweep["Train_Time_s"], "s--", color="#e69f00")
    ax2.set_ylabel("Training time (s)", color="#e69f00")
    ax.legend(loc="lower right"); ax.set_title("F1-score vs number of features")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "03_f1_vs_num_features.png", dpi=150); plt.close()

    return selected, rank, sweep


if __name__ == "__main__":
    from data_prep import load_processed
    run_feature_selection(load_processed())
