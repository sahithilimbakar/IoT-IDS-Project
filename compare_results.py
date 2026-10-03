"""
Before / after feature selection comparison: tables, graphs, detector bundles, summary report.
"""
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import config


def compare(full_res, red_res, data, full_models, red_models, selected):
    n_full, n_red = len(data["feature_names"]), len(selected)
    m = full_res.merge(red_res, on="Model", suffixes=("_Full", "_Reduced"))
    comp = pd.DataFrame(dict(Model=m["Model"], Features_Full=n_full, Features_Reduced=n_red))
    for c in ["Accuracy", "Precision", "Recall", "F1"]:
        comp[f"{c}_Full"] = m[f"{c}_Full"]; comp[f"{c}_Reduced"] = m[f"{c}_Reduced"]
        comp[f"{c}_Change"] = (m[f"{c}_Reduced"] - m[f"{c}_Full"]).round(5)
    for c, nm in [("Train_Time_s", "Train_Time_s"), ("Inference_us_per_sample", "Inference_us"),
                  ("Model_Size_MB", "Model_Size_MB")]:
        comp[f"{nm}_Full"] = m[f"{c}_Full"]; comp[f"{nm}_Reduced"] = m[f"{c}_Reduced"]
    comp["Train_Speedup_x"] = (m["Train_Time_s_Full"] / m["Train_Time_s_Reduced"].clip(lower=1e-9)).round(2)
    comp["Inference_Speedup_x"] = (m["Inference_us_per_sample_Full"] /
                                   m["Inference_us_per_sample_Reduced"].clip(lower=1e-9)).round(2)
    comp.to_csv(config.TABLES_DIR / "comparison_before_after.csv", index=False)

    # ---- grouped bar charts of the 4 metrics ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    x = np.arange(len(comp)); w = 0.38
    for ax, c in zip(axes, ["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(x - w/2, comp[f"{c}_Full"], w, label=f"All ({n_full})", color="#3b6fb6")
        ax.bar(x + w/2, comp[f"{c}_Reduced"], w, label=f"Selected ({n_red})", color="#e69f00")
        lo = min(comp[f"{c}_Full"].min(), comp[f"{c}_Reduced"].min())
        ax.set_ylim(max(0, lo - 0.05), 1.005)
        ax.set_xticks(x); ax.set_xticklabels([s.replace(" ", "\n") for s in comp["Model"]], fontsize=8)
        ax.set_title(c)
    axes[0].legend(fontsize=8)
    fig.suptitle("Performance before vs after feature selection (unseen test data)")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "04_metrics_before_after.png", dpi=150); plt.close()

    # ---- computational comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (c, t) in zip(axes, [("Train_Time_s", "Training time (s)"),
                                 ("Inference_us", "Inference time (us / sample)"),
                                 ("Model_Size_MB", "Model size (MB)")]):
        ax.bar(x - w/2, comp[f"{c}_Full"], w, label="All features", color="#3b6fb6")
        ax.bar(x + w/2, comp[f"{c}_Reduced"], w, label="Selected features", color="#e69f00")
        ax.set_xticks(x); ax.set_xticklabels([s.replace(" ", "\n") for s in comp["Model"]], fontsize=8)
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "05_computation_before_after.png", dpi=150); plt.close()

    # ---- best models + ROC ----
    best_full = full_res.sort_values("F1", ascending=False).iloc[0]["Model"]
    best_red = red_res.sort_values("F1", ascending=False).iloc[0]["Model"]
    ev_f = joblib.load(config.PROCESSED_DIR / "eval_full.joblib")
    ev_r = joblib.load(config.PROCESSED_DIR / "eval_reduced.joblib")
    y = data["y_test"]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for lab, ev, name, col in [(f"All {n_full} features", ev_f, best_full, "#3b6fb6"),
                               (f"Selected {n_red} features", ev_r, best_red, "#e69f00")]:
        fpr, tpr, _ = roc_curve(y, ev[name]["y_score"])
        ax.plot(fpr, tpr, color=col, label=f"{lab} - {name}")
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (best model)"); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "06_roc_best_models.png", dpi=150); plt.close()

    # ---- detector bundles (used by detector.py) ----
    for mode, models, name, feats in [("full", full_models, best_full, data["feature_names"]),
                                      ("reduced", red_models, best_red, selected)]:
        joblib.dump(dict(model=models[name], features=list(feats), medians=data["medians"],
                         model_name=name, mode=mode),
                    config.MODELS_DIR / f"ids_detector_{mode}.joblib", compress=3)

    write_summary(comp, full_res, red_res, best_full, best_red, selected, n_full, data)
    print("\n--- Before vs After feature selection ---")
    show = ["Model", "Features_Full", "Features_Reduced", "F1_Full", "F1_Reduced", "F1_Change",
            "Train_Speedup_x", "Inference_Speedup_x"]
    print(comp[show].to_string(index=False))
    return comp, best_full, best_red


def write_summary(comp, full_res, red_res, best_full, best_red, selected, n_full, data):
    rf = full_res.set_index("Model").loc[best_full]
    rr = red_res.set_index("Model").loc[best_red]
    ds = pd.read_csv(config.TABLES_DIR / "dataset_summary.csv").set_index("Item")["Value"]
    rank = pd.read_csv(config.TABLES_DIR / "feature_ranking.csv")
    pct = (1 - len(selected) / n_full) * 100
    L = []
    L.append("# Experimental Results Summary (auto-generated)\n")
    L.append("## Dataset")
    L.append(f"- Rows used: {int(ds['Rows used']):,} (train {int(ds['Training rows']):,} / unseen test {int(ds['Test rows (unseen)']):,})")
    L.append(f"- NORMAL: {int(ds['NORMAL rows']):,} | ATTACK: {int(ds['ATTACK rows']):,} | distinct labels: {int(ds['Distinct traffic labels'])}")
    L.append(f"- Features before selection: {n_full} | after selection: {len(selected)} ({pct:.1f}% reduction)\n")
    L.append("## Selected features (ranked by Random Forest importance)")
    for i, f in enumerate(selected, 1):
        L.append(f"{i}. {f}")
    L.append("\n## Best model on unseen test data")
    L.append("| Feature set | Model | #Features | Accuracy | Precision | Recall | F1 | Train time (s) | Inference (us/sample) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lab, r, nm in [("All features", rf, best_full), ("Selected features", rr, best_red)]:
        L.append(f"| {lab} | {nm} | {int(r['N_Features'])} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | "
                 f"{r['Recall']:.4f} | {r['F1']:.4f} | {r['Train_Time_s']:.2f} | {r['Inference_us_per_sample']:.2f} |")
    L.append("\n## All models: before vs after (F1-score)")
    L.append("| Model | F1 (all) | F1 (selected) | Change | Train speed-up | Inference speed-up |")
    L.append("|---|---|---|---|---|---|")
    for _, r in comp.iterrows():
        L.append(f"| {r['Model']} | {r['F1_Full']:.4f} | {r['F1_Reduced']:.4f} | {r['F1_Change']:+.4f} | "
                 f"{r['Train_Speedup_x']:.2f}x | {r['Inference_Speedup_x']:.2f}x |")
    L.append("\n## Key sentence for the paper (fill from your real numbers)")
    L.append(f"Reducing the feature set from {n_full} to {len(selected)} features ({pct:.1f}% fewer) changed the best "
             f"model's F1-score from {rf['F1']:.4f} to {rr['F1']:.4f} (change {rr['F1']-rf['F1']:+.4f}), "
             f"while training time went from {rf['Train_Time_s']:.1f}s to {rr['Train_Time_s']:.1f}s.")
    L.append("\n## Files")
    L.append("- results/tables/*.csv  - all numeric tables\n- results/figures/*.png - all graphs\n- models/ - trained models and detector bundles")
    (config.RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
