"""
STEP 3 : Train the Machine Learning models.
Models: Decision Tree, Random Forest, Hist Gradient Boosting, Logistic Regression.
"""
import re
import time
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config

warnings.filterwarnings("ignore")


def get_models():
    rs = config.RANDOM_STATE
    return {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=rs),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, class_weight="balanced",
                                                                 random_state=rs),
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=300, class_weight="balanced")),
    }


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def train_all(X_train, y_train, tag):
    """Train every model on X_train. tag = 'full' or 'reduced'. Returns (models, train_info_df)."""
    config.ensure_dirs()
    trained, rows = {}, []
    for name, model in get_models().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        dt = time.perf_counter() - t0
        path = config.MODELS_DIR / f"{tag}_{slug(name)}.joblib"
        joblib.dump(model, path, compress=3)
        trained[name] = model
        rows.append(dict(Model=name, Train_Time_s=round(dt, 3),
                         Model_Size_MB=round(path.stat().st_size / 1e6, 3)))
        print(f"  trained {name:<24} {dt:8.2f} s   ({X_train.shape[1]} features)")
    return trained, pd.DataFrame(rows)


if __name__ == "__main__":
    from data_prep import load_processed
    d = load_processed()
    print("STEP 3: TRAINING MODELS (all features)")
    train_all(d["X_train"], d["y_train"], "full")
