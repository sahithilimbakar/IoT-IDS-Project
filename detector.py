"""
STEP 4 : The IDS detector (prediction pipeline).

Takes raw network-flow data and predicts NORMAL or ATTACK for every row.

Usage:
    python detector.py --input my_traffic.csv --mode reduced
    python detector.py --demo 10                      # predict 10 random unseen test rows
    (mode = 'reduced' uses the selected features, 'full' uses all features)

In Python:
    from detector import IDSDetector
    ids = IDSDetector.load("reduced")
    result = ids.predict(dataframe)      # adds 'Prediction' and 'Attack_Probability'
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


class IDSDetector:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.medians = bundle["medians"]
        self.model_name = bundle["model_name"]
        self.mode = bundle["mode"]

    @classmethod
    def load(cls, mode="reduced"):
        path = MODELS_DIR / f"ids_detector_{mode}.joblib"
        if not path.exists():
            raise SystemExit(f"[ERROR] {path} not found. Run run_all.py first.")
        return cls(joblib.load(path))

    def _prepare(self, df):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            warnings.warn(f"Missing columns filled with training medians: {missing}")
        X = pd.DataFrame(index=df.index)
        for f in self.features:
            col = pd.to_numeric(df[f], errors="coerce") if f in df.columns else pd.Series(np.nan, index=df.index)
            X[f] = col
        X = X.replace([np.inf, -np.inf], np.nan).fillna(self.medians[self.features]).astype(np.float32)
        return X

    def predict(self, df):
        X = self._prepare(df)
        pred = self.model.predict(X)
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X)[:, 1]
        else:
            prob = np.full(len(X), np.nan)
        out = df.copy()
        out["Prediction"] = np.where(pred == 1, "ATTACK", "NORMAL")
        out["Attack_Probability"] = np.round(prob, 4)
        return out

    def predict_one(self, row_dict):
        return self.predict(pd.DataFrame([row_dict])).iloc[0][["Prediction", "Attack_Probability"]].to_dict()


def main():
    ap = argparse.ArgumentParser(description="IoT IDS - predict NORMAL or ATTACK")
    ap.add_argument("--input", help="CSV file with network traffic features")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--mode", choices=["reduced", "full"], default="reduced")
    ap.add_argument("--demo", type=int, default=0, help="predict N random rows from the unseen test set")
    a = ap.parse_args()
    ids = IDSDetector.load(a.mode)
    print(f"Loaded IDS: {ids.model_name} | mode={ids.mode} | features used={len(ids.features)}")

    if a.demo:
        data = joblib.load(BASE_DIR / "results" / "processed" / "processed_data.joblib")
        idx = np.random.RandomState(1).choice(len(data["X_test"]), a.demo, replace=False)
        df = data["X_test"].iloc[idx].reset_index(drop=True)
        actual = np.where(data["y_test"][idx] == 1, "ATTACK", "NORMAL")
        res = ids.predict(df)
        show = pd.DataFrame(dict(Actual=actual, Predicted=res["Prediction"],
                                 Attack_Probability=res["Attack_Probability"],
                                 Original_Label=data["attack_test"].iloc[idx].values))
        print(show.to_string(index=False))
        print(f"Correct: {(show.Actual == show.Predicted).sum()} / {len(show)}")
        return
    if not a.input:
        ap.error("give --input file.csv or --demo N")
    df = pd.read_csv(a.input, low_memory=False)
    res = ids.predict(df)
    res.to_csv(a.output, index=False)
    vc = res["Prediction"].value_counts()
    print(f"Predictions saved to {a.output}\n{vc.to_string()}")


if __name__ == "__main__":
    main()
