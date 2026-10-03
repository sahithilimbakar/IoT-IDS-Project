"""
STEP 1 + STEP 2 : Dataset loading and data preparation.

- finds every CSV under DATASET_DIR (extracts a .zip automatically if no CSV is found)
- samples each file, removes duplicates, fixes NaN / inf values
- converts the multi-class label into a binary label: 0 = NORMAL, 1 = ATTACK
- splits into train / test (stratified) and saves everything for the next steps
"""
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config


def find_csv_files(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise SystemExit(f"[ERROR] Dataset folder not found: {data_dir}\n"
                         f"        Put your CSV files (or the MERGED_CSV folder) inside it.")
    def scan():
        return sorted(p for p in data_dir.rglob("*.csv") if "__MACOSX" not in str(p))
    files = scan()
    if not files:
        for z in data_dir.glob("*.zip"):
            print(f"[INFO] No CSV found - extracting {z.name} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(data_dir)
        files = scan()
    if not files:
        raise SystemExit(f"[ERROR] No .csv files found under {data_dir}")
    return files


def find_label_col(columns):
    lowered = {str(c).strip().lower(): c for c in columns}
    for cand in config.LABEL_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    print("[WARN] No standard label column name found - using the LAST column as label.")
    return list(columns)[-1]


def load_sampled(files, frac):
    parts = []
    total = 0
    for i, f in enumerate(files, 1):
        n_file = 0
        for chunk in pd.read_csv(f, chunksize=200_000, low_memory=False):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            n_file += len(chunk)
            if frac < 1.0:
                chunk = chunk.sample(frac=frac, random_state=config.RANDOM_STATE)
            parts.append(chunk)
        total += n_file
        print(f"  [{i:>3}/{len(files)}] {f.name:<30} rows in file: {n_file:>9,}")
    df = pd.concat(parts, ignore_index=True)
    return df, total


def prepare():
    config.ensure_dirs()
    print("=" * 70)
    print("STEP 1-2: LOADING AND PREPARING DATASET")
    print("=" * 70)
    files = find_csv_files(config.DATASET_DIR)
    print(f"Found {len(files)} CSV file(s) in {config.DATASET_DIR}")
    df, total_rows = load_sampled(files, config.SAMPLE_FRAC)
    n_sampled = len(df)
    print(f"Total rows in dataset: {total_rows:,} | sampled: {n_sampled:,} "
          f"({config.SAMPLE_FRAC*100:.2f}% per file)")

    label_col = find_label_col(df.columns)
    print(f"Label column: '{label_col}'")

    df = df.dropna(subset=[label_col]).drop_duplicates().reset_index(drop=True)
    n_dedup = len(df)
    print(f"After removing duplicates / missing labels: {n_dedup:,} rows")

    attack_type = df[label_col].astype(str).str.strip()
    y = (~attack_type.str.lower().isin(config.NORMAL_NAMES)).astype(int).values   # 0 normal, 1 attack
    if len(np.unique(y)) < 2:
        raise SystemExit("[ERROR] Only one class found (all NORMAL or all ATTACK). "
                         "Increase SAMPLE_FRAC or check NORMAL_NAMES in config.py")

    # ---- features: numeric only, clean NaN / inf, drop constant columns ----
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"Dropping non-numeric / empty columns: {all_nan}")
        X = X.drop(columns=all_nan)
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant:
        print(f"Dropping constant columns: {constant}")
        X = X.drop(columns=constant)
    medians = X.median()
    X = X.fillna(medians).astype(np.float32)
    print(f"Features kept: {X.shape[1]}")

    # ---- optional stratified cap on dataset size ----
    idx = np.arange(len(X))
    if len(X) > config.MAX_ROWS:
        idx, _ = train_test_split(idx, train_size=config.MAX_ROWS, stratify=y,
                                  random_state=config.RANDOM_STATE)
        X, y, attack_type = X.iloc[idx], y[idx], attack_type.iloc[idx]
        print(f"Capped to {len(X):,} rows (stratified).")
    X = X.reset_index(drop=True)
    attack_type = attack_type.reset_index(drop=True)

    # ---- train / test split (test = completely unseen data) ----
    X_train, X_test, y_train, y_test, at_train, at_test = train_test_split(
        X, y, attack_type, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)

    data = dict(X_train=X_train.reset_index(drop=True), X_test=X_test.reset_index(drop=True),
                y_train=y_train, y_test=y_test,
                attack_train=at_train.reset_index(drop=True), attack_test=at_test.reset_index(drop=True),
                feature_names=list(X.columns), medians=medians[X.columns])
    joblib.dump(data, config.PROCESSED_DIR / "processed_data.joblib", compress=3)

    # ---- summary table + figure ----
    n_normal, n_attack = int((y == 0).sum()), int((y == 1).sum())
    summary = pd.DataFrame([
        ("CSV files", len(files)), ("Rows in original dataset", total_rows),
        ("Rows sampled", n_sampled), ("Rows after de-duplication", n_dedup),
        ("Rows used", len(X)), ("Number of features", X.shape[1]),
        ("NORMAL rows", n_normal), ("ATTACK rows", n_attack),
        ("Distinct traffic labels", attack_type.nunique()),
        ("Training rows", len(X_train)), ("Test rows (unseen)", len(X_test)),
    ], columns=["Item", "Value"])
    summary.to_csv(config.TABLES_DIR / "dataset_summary.csv", index=False)
    dist = attack_type.value_counts().rename_axis("Label").reset_index(name="Count")
    dist.to_csv(config.TABLES_DIR / "class_distribution.csv", index=False)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].bar(["NORMAL", "ATTACK"], [n_normal, n_attack], color=["#2e9e4f", "#d64545"])
    for i, v in enumerate([n_normal, n_attack]):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    ax[0].set_title("Normal vs Attack traffic"); ax[0].set_ylabel("Number of samples")
    top = dist.head(15).iloc[::-1]
    ax[1].barh(top["Label"], top["Count"], color="#3b6fb6")
    ax[1].set_xscale("log"); ax[1].set_title("Top traffic labels (log scale)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "01_class_distribution.png", dpi=150)
    plt.close()
    return data


def load_processed():
    p = config.PROCESSED_DIR / "processed_data.joblib"
    if not p.exists():
        raise SystemExit("[ERROR] Processed data not found. Run: python data_prep.py  (or run_all.py)")
    return joblib.load(p)


if __name__ == "__main__":
    prepare()
