"""
Run the COMPLETE project end to end:

    Dataset -> Data Preparation -> Train -> Detector -> Test -> Feature Selection -> Re-test -> Compare

Usage:
    python run_all.py                       # normal run (1% of each CSV, max 300k rows)
    python run_all.py --quick               # fast test run (0.3% sample, max 80k rows)
    python run_all.py --frac 0.02 --max-rows 500000      # bigger run, better results
    python run_all.py --data "E:/IoT_IDS_Project/dataset"  # different dataset folder
    python run_all.py --k 10                # force exactly 10 selected features
    python run_all.py --skip-prep           # re-use already prepared data
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="dataset folder (default: ./dataset)")
    ap.add_argument("--frac", type=float, help="fraction of each CSV to sample (default 0.01)")
    ap.add_argument("--max-rows", type=int, help="max rows used (default 300000)")
    ap.add_argument("--k", type=int, help="force the number of selected features")
    ap.add_argument("--quick", action="store_true", help="small fast run for testing")
    ap.add_argument("--skip-prep", action="store_true")
    a = ap.parse_args()

    if a.quick:
        config.SAMPLE_FRAC, config.MAX_ROWS = 0.003, 80_000
    if a.data: config.DATASET_DIR = Path(a.data)
    if a.frac: config.SAMPLE_FRAC = a.frac
    if a.max_rows: config.MAX_ROWS = a.max_rows
    if a.k: config.FORCE_K = a.k

    # import after config overrides (modules read config.X at call time anyway)
    import data_prep, train_models, evaluate, feature_selection, compare_results
    from detector import IDSDetector

    t_all = time.perf_counter()
    config.ensure_dirs()
    data = data_prep.load_processed() if a.skip_prep else data_prep.prepare()
    Xtr, Xte, ytr, yte = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    print("\n" + "=" * 70 + "\nSTEP 3 + 5: TRAIN AND TEST (ALL FEATURES)\n" + "=" * 70)
    full_models, full_info = train_models.train_all(Xtr, ytr, "full")
    full_res, _ = evaluate.evaluate_models(full_models, full_info, Xte, yte, data["attack_test"], "full")

    selected, rank, sweep = feature_selection.run_feature_selection(data)

    print("\n" + "=" * 70 + f"\nSTEP 6b: RE-TRAIN AND RE-TEST ({len(selected)} SELECTED FEATURES)\n" + "=" * 70)
    red_models, red_info = train_models.train_all(Xtr[selected], ytr, "reduced")
    red_res, _ = evaluate.evaluate_models(red_models, red_info, Xte[selected], yte,
                                          data["attack_test"], "reduced")

    compare_results.compare(full_res, red_res, data, full_models, red_models, selected)

    print("\n" + "=" * 70 + "\nSTEP 4: DETECTOR DEMO (10 unseen samples)\n" + "=" * 70)
    ids = IDSDetector.load("reduced")
    idx = np.random.RandomState(1).choice(len(Xte), 10, replace=False)
    res = ids.predict(Xte.iloc[idx].reset_index(drop=True))
    demo = pd.DataFrame(dict(Actual=np.where(yte[idx] == 1, "ATTACK", "NORMAL"),
                             Predicted=res["Prediction"], Attack_Probability=res["Attack_Probability"],
                             Original_Label=data["attack_test"].iloc[idx].values))
    demo.to_csv(config.TABLES_DIR / "sample_predictions.csv", index=False)
    print(demo.to_string(index=False))

    print(f"\nDONE in {(time.perf_counter()-t_all)/60:.1f} min.")
    print(f"Results : {config.RESULTS_DIR}\n  figures/  tables/  RESULTS_SUMMARY.md\nModels  : {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
