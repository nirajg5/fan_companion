
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "win_probability_dataset.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "win_probability.joblib"

FEATURES = [
    "runs",
    "wickets",
    "legal_balls",
    "balls_remaining",
    "target",
    "runs_required",
    "run_rate",
    "required_run_rate",
]
TARGET = "batting_team_won"
GROUP = "match_id"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}. "
            "Run the dataset preparation script first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET, GROUP]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=[TARGET, GROUP]).copy()
    df[TARGET] = df[TARGET].astype(int)

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Training requires both winning and losing examples."
        )

    if df[GROUP].nunique() < 5:
        raise ValueError(
            "At least five distinct matches are required for "
            "a preliminary grouped split."
        )

    # Keep every match entirely in either train or test.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(df[FEATURES], df[TARGET], groups=df[GROUP])
    )

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\nEvaluation on held-out matches")
    print("Training matches:", train_df[GROUP].nunique())
    print("Testing matches:", test_df[GROUP].nunique())
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))
    print(
        "Log loss:",
        round(log_loss(y_test, probabilities, labels=[0, 1]), 4),
    )

    if y_test.nunique() == 2:
        print(
            "ROC-AUC:",
            round(roc_auc_score(y_test, probabilities), 4),
        )
    else:
        print("ROC-AUC: unavailable; test set contains one class.")

    # Save metadata with the model so inference uses the same features.
    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "positive_class": "chasing_team_wins",
        "training_matches": sorted(
            train_df[GROUP].astype(str).unique().tolist()
        ),
        "test_matches": sorted(
            test_df[GROUP].astype(str).unique().tolist()
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
