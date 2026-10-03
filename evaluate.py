"""
STEP 5 : Test the system on unseen data and measure accuracy, precision, recall, F1.
Also measures inference time and saves confusion matrices.
"""
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = dict(Accuracy=accuracy_score(y_true, y_pred),
               Precision=precision_score(y_true, y_pred, zero_division=0),
               Recall=recall_score(y_true, y_pred, zero_division=0),
               F1=f1_score(y_true, y_pred, zero_division=0),
               False_Alarm_Rate=fp / max(fp + tn, 1))
    if y_score is not None:
        try:
            out["ROC_AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            out["ROC_AUC"] = np.nan
    return out


def evaluate_models(models, train_info, X_test, y_test, attack_test, tag):
    """Evaluate all models on the unseen test set. Saves tables + figures tagged with `tag`."""
    rows, store, cms = [], {}, {}
    for name, m in models.items():
        t0 = time.perf_counter()
        pred = m.predict(X_test)
        dt = time.perf_counter() - t0
        score = get_score(m, X_test)
        met = compute_metrics(y_test, pred, score)
        met.update(Model=name, N_Features=X_test.shape[1],
                   Inference_us_per_sample=dt / len(X_test) * 1e6)
        rows.append(met)
        store[name] = dict(y_pred=pred, y_score=score)
        cms[name] = confusion_matrix(y_test, pred, labels=[0, 1])
    res = pd.DataFrame(rows).merge(train_info, on="Model")
    cols = ["Model", "N_Features", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC",
            "False_Alarm_Rate", "Train_Time_s", "Inference_us_per_sample", "Model_Size_MB"]
    res = res[cols].round(5)
    res.to_csv(config.TABLES_DIR / f"metrics_{tag}_features.csv", index=False)
    joblib.dump(store, config.PROCESSED_DIR / f"eval_{tag}.joblib")

    # confusion matrices
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["NORMAL", "ATTACK"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["NORMAL", "ATTACK"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle(f"Confusion matrices - {tag} feature set", fontsize=12)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"confusion_matrices_{tag}.png", dpi=150)
    plt.close()

    # per attack type detection rate for the best model
    best = res.sort_values("F1", ascending=False).iloc[0]["Model"]
    pred = store[best]["y_pred"]
    d = pd.DataFrame(dict(Label=attack_test.values, pred=pred))
    d = d[~d["Label"].str.lower().isin(config.NORMAL_NAMES)]
    per = (d.groupby("Label")["pred"].agg(["mean", "size"]).rename(
        columns={"mean": "Detection_Rate", "size": "Test_Samples"}).sort_values("Detection_Rate"))
    per.round(5).to_csv(config.TABLES_DIR / f"per_attack_detection_{tag}.csv")

    print(f"\n--- Test results ({tag} features, {X_test.shape[1]} features) ---")
    print(res.to_string(index=False))
    print(f"Best model ({tag}): {best}")
    return res, best


if __name__ == "__main__":
    from data_prep import load_processed
    from train_models import train_all
    d = load_processed()
    models, info = train_all(d["X_train"], d["y_train"], "full")
    evaluate_models(models, info, d["X_test"], d["y_test"], d["attack_test"], "full")
