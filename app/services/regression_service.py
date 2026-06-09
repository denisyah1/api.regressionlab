import math
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.utils.csv_loader import load_csv
from app.utils.data_cleaning import clean_dataframe
from app.utils.feature_detection import detect_feature_types
from app.services.preprocessing import build_preprocessor
from app.services.model_factory import get_regression_models
from app.utils.json_sanitizer import sanitize
from app.core.config import DEFAULT_NULL_STRATEGY, TRAIN_TEST_SPLIT_RATIO


def _train(file, target_column, feature_columns, null_strategy, drop_columns=None):
    """Shared core logic: load → clean → train. Returns raw results dict."""

    df = load_csv(file)

    # Normalize column names
    original_cols = list(df.columns)
    stripped_cols = [c.strip() if isinstance(c, str) else c for c in original_cols]
    if len(set(stripped_cols)) != len(stripped_cols):
        raise HTTPException(400, "Duplicate columns detected after trimming spaces")
    if stripped_cols != original_cols:
        df.columns = stripped_cols
    if target_column:
        target_column = target_column.strip()
    if feature_columns:
        feature_columns = [c.strip() for c in feature_columns]

    if not feature_columns:
        raise HTTPException(400, "Feature columns cannot be empty")
    if target_column not in df.columns:
        raise HTTPException(400, f"Target column '{target_column}' not found")
    for col in feature_columns:
        if col not in df.columns:
            raise HTTPException(400, f"Feature column '{col}' not found")

    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")

    strategy = null_strategy or DEFAULT_NULL_STRATEGY
    df = clean_dataframe(df, feature_columns, target_column, strategy)

    X = df[feature_columns]
    y = df[target_column]

    if not pd.api.types.is_numeric_dtype(y):
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' must be numeric for regression"
        )

    numeric_features, categorical_features = detect_feature_types(df, feature_columns)
    if not numeric_features and not categorical_features:
        raise HTTPException(400, "No valid features detected")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=42
    )
    if len(y_test) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough test samples to evaluate regression. Please provide more data."
        )

    models = get_regression_models(preprocessor)
    best_model = None
    best_model_name = None
    best_r2 = -1e9
    best_test_pred = None
    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            if math.isnan(test_r2):
                test_r2 = None
            test_mse = mean_squared_error(y_test, test_pred)
            results[name] = {
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_mse": test_mse
            }
            if test_r2 is not None and test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model
                best_model_name = name
                best_test_pred = test_pred
        except Exception as e:
            results[name] = {"error": str(e)}

    if not best_model:
        raise HTTPException(500, "All models failed")

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "results": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_test_pred": best_test_pred,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "df": df,
        "strategy": strategy,
    }


def run_regression(file, target_column=None, feature_columns=None,
                   null_strategy=None, drop_columns=None):
    """
    Runs regression and returns all results including plot_data inline.
    plot_data is NO LONGER stored in a shared dict (PLOT_STORE removed).
    It is built as a local variable and embedded directly in the response.
    """
    t = _train(file, target_column, feature_columns, null_strategy, drop_columns)

    # Build plot data locally — never stored globally
    plot_data = {
        "train": {
            "y_actual": t["y_train"].tolist(),
            "y_pred": t["best_model"].predict(t["X_train"]).tolist()
        },
        "test": {
            "y_actual": t["y_test"].tolist(),
            "y_pred": t["best_test_pred"].tolist()
        }
    }

    response = {
        "best_model": t["best_model_name"],
        "feature_engineering": {
            "numeric_features": t["numeric_features"],
            "categorical_features": t["categorical_features"],
        },
        "data_info": {
            "rows": len(t["df"]),
            "train_rows": len(t["X_train"]),
            "test_rows": len(t["X_test"]),
            "null_strategy": t["strategy"],
        },
        "model_comparison": t["results"],
        # plot_data is embedded here — frontend reads response.plot_data directly,
        # no need for a separate GET /api/regression/plot request
        "plot_data": plot_data,
    }
    return sanitize(response)


def run_regression_and_get_model(file, target_column=None,
                                  feature_columns=None, null_strategy=None):
    """
    Used only by POST /api/model/download.
    Re-runs training and returns the model object in memory for streaming.
    Model is NEVER saved to disk.
    """
    t = _train(file, target_column, feature_columns, null_strategy)
    return t["best_model"], t["best_model_name"]