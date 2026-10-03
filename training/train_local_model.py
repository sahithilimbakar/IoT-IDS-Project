from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent

BENIGN_FILE = (
    PROJECT_DIR
    / "local_data"
    / "benign"
    / "benign_features.csv"
)

ATTACK_FILE = (
    PROJECT_DIR
    / "local_data"
    / "attack"
    / "attack_features.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_MODEL = MODEL_DIR / "local_environment_model.joblib"
OUTPUT_RESULTS = PROJECT_DIR / "local_data" / "local_training_results.txt"

FEATURES = [
    "Number",
    "ack_flag_number",
    "HTTPS",
    "Tot size",
    "Rate",
]


def main():

    print("=" * 60)
    print("LOCAL ENVIRONMENT IDS TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # Check input files
    # ---------------------------------------------------------

    if not BENIGN_FILE.exists():
        print()
        print("ERROR: Benign feature file not found:")
        print(BENIGN_FILE)
        return

    if not ATTACK_FILE.exists():
        print()
        print("ATTACK FEATURE FILE NOT FOUND YET:")
        print(ATTACK_FILE)
        print()
        print("The attack PCAP still needs to be processed.")
        print("Do not train until attack_features.csv exists.")
        return

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print()
    print("Loading benign data...")
    benign = pd.read_csv(BENIGN_FILE)

    print(f"Benign windows: {len(benign)}")

    print()
    print("Loading attack data...")
    attack = pd.read_csv(ATTACK_FILE)

    print(f"Attack windows: {len(attack)}")

    # ---------------------------------------------------------
    # Add labels
    # ---------------------------------------------------------

    benign["Label"] = 0
    attack["Label"] = 1

    data = pd.concat(
        [benign, attack],
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Validate features
    # ---------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing:
        print()
        print("ERROR: Missing features:")
        print(missing)
        return

    # ---------------------------------------------------------
    # Remove invalid values
    # ---------------------------------------------------------

    data = data[FEATURES + ["Label"]].copy()

    data = data.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    data = data.dropna()

    # ---------------------------------------------------------
    # Display class distribution
    # ---------------------------------------------------------

    print()
    print("Class distribution:")
    print(
        data["Label"]
        .value_counts()
        .sort_index()
        .rename({
            0: "NORMAL",
            1: "ATTACK"
        })
    )

    # ---------------------------------------------------------
    # Features and labels
    # ---------------------------------------------------------

    X = data[FEATURES]
    y = data["Label"]

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # ---------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------

    print()
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("LOCAL IDS TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print()
    print("Confusion Matrix:")
    print(matrix)

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["NORMAL", "ATTACK"],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    medians = X_train.median()

    bundle = {
        "model": model,
        "features": FEATURES,
        "medians": medians,
        "model_name": "Random Forest",
        "mode": "local_environment",
    }

    joblib.dump(
        bundle,
        OUTPUT_MODEL
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "LOCAL ENVIRONMENT IDS RESULTS\n"
        )

        file.write(
            "==============================\n\n"
        )

        file.write(
            f"Benign windows: {len(benign)}\n"
        )

        file.write(
            f"Attack windows: {len(attack)}\n\n"
        )

        file.write(
            f"Training samples: {len(X_train)}\n"
        )

        file.write(
            f"Testing samples: {len(X_test)}\n\n"
        )

        file.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall   : {recall:.4f}\n"
        )

        file.write(
            f"F1 Score : {f1:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix:\n"
        )

        file.write(
            str(matrix)
        )

        file.write("\n\n")

        file.write(
            classification_report(
                y_test,
                predictions,
                target_names=["NORMAL", "ATTACK"],
                zero_division=0
            )
        )

    print()
    print("MODEL SAVED:")
    print(OUTPUT_MODEL)

    print()
    print("RESULTS SAVED:")
    print(OUTPUT_RESULTS)


if __name__ == "__main__":
    main()