"""Small, bounded helpers shared by the opt-in body upload analyzers."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score

from core.types import MetricsReport, Prediction, UploadAnalysis

MAX_ANALYZE_ROWS = 25_000
MAX_RETURNED_PREDICTIONS = 100


def read_csv_table(
    content: bytes,
    filename: str,
    *,
    module_id: str,
    expected_columns: Iterable[str],
    optional_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Parse a UTF-8 CSV and fail with a schema-oriented message, not a pandas traceback."""
    try:
        frame = pd.read_csv(BytesIO(content))
    except Exception as exc:  # noqa: BLE001 - bad user files should be a readable 422
        raise ValueError(f"Could not read {filename} as CSV: {type(exc).__name__}: {exc}") from None

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise ValueError(f"{module_id} CSV has duplicate column names: {', '.join(duplicates)}.")
    if frame.empty:
        raise ValueError(f"{module_id} CSV contains no data rows.")

    expected = list(expected_columns)
    optional = set(optional_columns)
    required = [column for column in expected if column not in optional]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        expected_text = ", ".join(expected)
        raise ValueError(
            f"{module_id} CSV is missing required columns: {', '.join(missing)}. "
            f"Expected columns: {expected_text}."
        )
    if len(frame) > MAX_ANALYZE_ROWS:
        raise ValueError(
            f"{module_id} CSV has {len(frame):,} rows; this analyzer is capped at "
            f"{MAX_ANALYZE_ROWS:,} rows per upload."
        )
    return frame


def coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce named columns, rejecting non-numeric text while allowing blank cells."""
    out = frame.copy()
    for column in columns:
        original = out[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Column '{column}' must contain numbers (blank cells are allowed).")
        if not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' contains a non-finite number.")
        out[column] = converted
    return out


def coerce_binary_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Accept 0/1, yes/no, and true/false for a binary feature or optional target."""
    out = frame.copy()
    original = out[column]
    normalized = original.astype("string").str.strip().str.lower()
    text_values = normalized.map({"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0})
    numeric = pd.to_numeric(original, errors="coerce")
    converted = numeric.where(numeric.notna(), text_values)
    invalid = original.notna() & converted.isna()
    if invalid.any() or not converted.dropna().isin([0, 1]).all():
        raise ValueError(f"Column '{column}' must contain only 0/1 or yes/no values.")
    out[column] = converted
    return out


def prediction_inputs(row: pd.Series, columns: Iterable[str]) -> dict[str, float]:
    """Return finite feature values in the Prediction contract, omitting blank cells."""
    inputs: dict[str, float] = {}
    for column in columns:
        value = row[column]
        if pd.notna(value):
            inputs[column] = float(value)
    return inputs


def classification_upload_result(
    *,
    module_id: str,
    filename: str,
    n_rows: int,
    predictions: list[Prediction],
    threshold: float,
    target_name: str,
    labels: list[int] | None,
    model_source: str,
) -> UploadAnalysis:
    """Summarize predicted scores, distinguishing actual labels from unlabeled inference."""
    scores_array = np.asarray([prediction.score for prediction in predictions], dtype=float)
    predicted = (scores_array >= threshold).astype(int)
    positive_rate = float(np.mean(labels)) if labels is not None else float(np.mean(predicted))
    scores = {
        "mean_score": float(np.mean(scores_array)),
        "median_score": float(np.median(scores_array)),
        "predicted_positive_rate": float(np.mean(predicted)),
    }
    confusion: dict[str, int] = {}
    notes = [
        f"Scored {len(predictions):,} uploaded rows with the existing {model_source} model; the model was not retrained.",
        "Upload data is processed in memory and is not saved.",
    ]
    if labels is None:
        notes.append(
            f"No '{target_name}' target column was supplied. positive_rate is the share predicted positive "
            "at the operating threshold, not observed outcome prevalence."
        )
    else:
        truth = np.asarray(labels, dtype=int)
        tn = int(np.sum((truth == 0) & (predicted == 0)))
        fp = int(np.sum((truth == 0) & (predicted == 1)))
        fn = int(np.sum((truth == 1) & (predicted == 0)))
        tp = int(np.sum((truth == 1) & (predicted == 1)))
        confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        scores.update(
            {
                "accuracy": float(accuracy_score(truth, predicted)),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
            }
        )
        if len(np.unique(truth)) > 1:
            scores["roc_auc"] = float(roc_auc_score(truth, scores_array))
            scores["pr_auc"] = float(average_precision_score(truth, scores_array))
        notes.append(f"Observed prevalence comes from the uploaded '{target_name}' labels.")

    return UploadAnalysis(
        module_id=module_id,
        filename=filename,
        n_rows=n_rows,
        metrics=MetricsReport(
            n_samples=n_rows,
            positive_rate=positive_rate,
            scores=scores,
            confusion=confusion,
            threshold=float(threshold),
            notes=notes,
        ),
        predictions=sorted(predictions, key=lambda item: item.score, reverse=True)[:MAX_RETURNED_PREDICTIONS],
        predictions_truncated=len(predictions) > MAX_RETURNED_PREDICTIONS,
    )
