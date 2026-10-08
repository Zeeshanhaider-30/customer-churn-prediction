"""Train, evaluate, and export the customer churn prediction pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path(__file__).resolve().parent / "Telco_customer_churn.xlsx"
MODEL_PATH = Path(__file__).resolve().parent / "best_model.joblib"
TARGET_COLUMN = "Churn Value"
DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]
NUMERICAL_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]


def build_pipeline(numerical_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Create a model pipeline that learns preprocessing only from its training fold."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LogisticRegression())]
    )


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load source features and target while excluding identifiers and target leakage."""
    data = pd.read_excel(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET_COLUMN!r}.")

    data = data.drop(columns=[column for column in DROP_COLUMNS if column in data.columns])
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN].astype(int)
    return features, target


def train_model(
    data_path: Path = DATA_PATH,
) -> tuple[Pipeline, str, pd.DataFrame, dict[str, object]]:
    """Tune candidates with stratified CV, evaluate on a held-out test set, and refit."""
    features, target = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    numerical_features = [column for column in NUMERICAL_FEATURES if column in x_train.columns]
    categorical_features = [
        column for column in x_train.columns if column not in numerical_features
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logistic = build_pipeline(numerical_features, categorical_features)
    logistic_scores = cross_val_score(
        logistic,
        x_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    logistic.fit(x_train, y_train)

    random_forest = build_pipeline(numerical_features, categorical_features)
    random_forest.set_params(
        model=RandomForestClassifier(random_state=42, class_weight="balanced")
    )
    random_forest_search = GridSearchCV(
        random_forest,
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 12],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    random_forest_search.fit(x_train, y_train)

    xgboost = build_pipeline(numerical_features, categorical_features)
    xgboost.set_params(
        model=XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    )
    xgboost_search = GridSearchCV(
        xgboost,
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    xgboost_search.fit(x_train, y_train)

    candidates = {
        "Logistic Regression": (float(logistic_scores.mean()), logistic),
        "Random Forest": (
            float(random_forest_search.best_score_),
            random_forest_search.best_estimator_,
        ),
        "XGBoost": (float(xgboost_search.best_score_), xgboost_search.best_estimator_),
    }
    best_model_name = max(candidates, key=lambda name: candidates[name][0])
    best_cv_model = candidates[best_model_name][1]
    predictions = best_cv_model.predict(x_test)
    churn_probabilities = best_cv_model.predict_proba(x_test)[:, 1]

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, churn_probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    results = pd.DataFrame(
        [
            {
                "Model": name,
                "Mean CV F1": score,
                "Best Parameters": (
                    random_forest_search.best_params_
                    if name == "Random Forest"
                    else xgboost_search.best_params_
                    if name == "XGBoost"
                    else "LogisticRegression defaults"
                ),
            }
            for name, (score, _) in candidates.items()
        ]
    ).sort_values("Mean CV F1", ascending=False)

    deployment_model = build_pipeline(numerical_features, categorical_features)
    deployment_model.set_params(model=best_cv_model.named_steps["model"])
    deployment_model.fit(features, target)
    return deployment_model, best_model_name, results, metrics


def main() -> None:
    model, best_model_name, results, metrics = train_model()
    joblib.dump(model, MODEL_PATH)
    print("Cross-validation results:")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print("Held-out test metrics:")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{metric.replace('_', ' ').title()}: {metrics[metric]:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; order: Stay, Churn):")
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Stay", "Churn"],
        columns=["Stay", "Churn"],
    )
    print(confusion)
    print(f"\nSaved full preprocessing/model pipeline to {MODEL_PATH.name}.")


if __name__ == "__main__":
    main()
