from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.dataset_loader import HF_DATASET_ID, load_dataset
from utils.preprocessing import (
    analyze_urls,
    count_email_addresses,
    count_urls,
    detect_suspicious_keywords,
    phishing_pattern_categories,
    preprocess_text,
    security_indicators,
    word_count,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
MODEL_COLLECTION_PATH = MODEL_DIR / "models.pkl"
EXPLAINERS_PATH = MODEL_DIR / "explainers.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
ARTIFACT_VERSION = 2

LABELS = ["Legitimate", "Phishing"]
FINAL_MODEL_NAME = "Calibrated Logistic Regression"
DEFAULT_THRESHOLD = 0.5


@dataclass
class ModelBundle:
    model: Any
    vectorizer: TfidfVectorizer
    metadata: dict[str, Any]
    dataset: pd.DataFrame
    models: dict[str, Any]
    explainers: dict[str, Any]


def _prefer_huggingface_dataset() -> bool:
    return os.getenv("PHISHGUARD_USE_HF", "true").strip().lower() not in {"0", "false", "no"}


def _current_dataset(force_refresh: bool = False) -> pd.DataFrame:
    dataset = load_dataset(prefer_huggingface=_prefer_huggingface_dataset(), force_refresh=force_refresh)
    if dataset.empty:
        raise ValueError("Dataset tidak memiliki label Legitimate/Phishing yang valid.")
    return dataset


def _build_models() -> dict[str, Any]:
    return {
        "Naive Bayes": MultinomialNB(),
        "Calibrated Logistic Regression": CalibratedClassifierCV(
            estimator=LogisticRegression(C=4.0, max_iter=1000, random_state=42),
            method="sigmoid",
            cv=3,
        ),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=42),
            method="sigmoid",
            cv=3,
        ),
    }


def _build_explainers() -> dict[str, Any]:
    return {
        "Naive Bayes": MultinomialNB(),
        "Calibrated Logistic Regression": LogisticRegression(C=4.0, max_iter=1000, random_state=42),
        "Calibrated Linear SVM": LinearSVC(random_state=42),
    }


def _prepare_features(train_text: pd.Series, test_text: pd.Series) -> tuple[TfidfVectorizer, Any, Any]:
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_text.map(preprocess_text))
    x_test = vectorizer.transform(test_text.map(preprocess_text))
    return vectorizer, x_train, x_test


def _phishing_scores(model: Any, features: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)
        phishing_index = classes.index("Phishing") if "Phishing" in classes else classes.index(1)
        return np.asarray(probabilities[:, phishing_index], dtype=float)

    decision = np.asarray(model.decision_function(features), dtype=float)
    if decision.ndim > 1:
        decision = decision[:, -1]
    return 1 / (1 + np.exp(-decision))


def _labels_from_scores(scores: np.ndarray, threshold: float) -> list[str]:
    return ["Phishing" if score >= threshold else "Legitimate" for score in scores]


def _threshold_rows(y_true_binary: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for threshold in np.arange(0.2, 0.81, 0.1):
        predictions = (scores >= threshold).astype(int)
        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Accuracy": round(float(accuracy_score(y_true_binary, predictions)), 4),
                "Precision": round(float(precision_score(y_true_binary, predictions, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_true_binary, predictions, zero_division=0)), 4),
                "F1-Score": round(float(f1_score(y_true_binary, predictions, zero_division=0)), 4),
            }
        )
    return rows


def _curve_rows(name: str, y_true_binary: np.ndarray, scores: np.ndarray) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], list[dict[str, float | str]]]:
    fpr, tpr, roc_thresholds = roc_curve(y_true_binary, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, scores)
    calibrated_true, calibrated_pred = calibration_curve(y_true_binary, scores, n_bins=10, strategy="uniform")

    roc_rows = [
        {
            "Model": name,
            "FPR": round(float(x), 4),
            "TPR": round(float(y), 4),
            "Threshold": round(float(threshold), 4) if np.isfinite(threshold) else 1.0,
        }
        for x, y, threshold in zip(fpr, tpr, roc_thresholds)
    ]
    pr_rows = [
        {
            "Model": name,
            "Recall": round(float(x), 4),
            "Precision": round(float(y), 4),
            "Threshold": round(float(pr_thresholds[min(index, len(pr_thresholds) - 1)]), 4) if len(pr_thresholds) else 1.0,
        }
        for index, (y, x) in enumerate(zip(precision, recall))
    ]
    calibration_rows = [
        {
            "Model": name,
            "Mean Predicted Probability": round(float(predicted), 4),
            "Observed Phishing Rate": round(float(observed), 4),
        }
        for observed, predicted in zip(calibrated_true, calibrated_pred)
    ]
    return roc_rows, pr_rows, calibration_rows


def _error_examples(test_df: pd.DataFrame, predictions: list[str], scores: np.ndarray, limit: int = 16) -> list[dict[str, Any]]:
    rows = []
    indexed_test = test_df.reset_index(drop=True)
    for row_index, (actual, predicted, score) in enumerate(zip(indexed_test["label"], predictions, scores)):
        if actual == predicted:
            continue
        text = str(indexed_test.loc[row_index, "text"])
        rows.append(
            {
                "Error Type": "False Positive" if actual == "Legitimate" and predicted == "Phishing" else "False Negative",
                "Actual": actual,
                "Predicted": predicted,
                "Phishing Probability": round(float(score), 4),
                "Snippet": " ".join(text.split())[:260],
            }
        )

    rows.sort(key=lambda item: abs(item["Phishing Probability"] - DEFAULT_THRESHOLD))
    return rows[:limit]


def _cross_validation_rows(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    processed_text = dataset["text"].map(preprocess_text)
    y_binary = dataset["label"].map({"Legitimate": 0, "Phishing": 1})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_specs = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(C=4.0, max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(random_state=42),
    }

    rows = []
    for name, estimator in model_specs.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("model", estimator),
            ]
        )
        scores = cross_validate(
            pipeline,
            processed_text,
            y_binary,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
        )
        rows.append(
            {
                "Model": name,
                "CV Accuracy Mean": round(float(scores["test_accuracy"].mean()), 4),
                "CV Accuracy Std": round(float(scores["test_accuracy"].std()), 4),
                "CV Precision Mean": round(float(scores["test_precision"].mean()), 4),
                "CV Recall Mean": round(float(scores["test_recall"].mean()), 4),
                "CV F1 Mean": round(float(scores["test_f1"].mean()), 4),
            }
        )
    return rows


def train_and_save(force_refresh: bool = False) -> ModelBundle:
    dataset = _current_dataset(force_refresh=force_refresh)
    train_df, test_df = train_test_split(
        dataset,
        test_size=0.3,
        random_state=42,
        stratify=dataset["label"],
    )

    comparison_rows = []
    confusion_by_model: dict[str, list[list[int]]] = {}
    threshold_analysis: dict[str, list[dict[str, float]]] = {}
    roc_rows_all: list[dict[str, float | str]] = []
    pr_rows_all: list[dict[str, float | str]] = []
    calibration_rows_all: list[dict[str, float | str]] = []
    error_analysis: list[dict[str, Any]] = []

    vectorizer, x_train, x_test = _prepare_features(train_df["text"], test_df["text"])
    y_train = train_df["label"]
    y_test = test_df["label"]
    y_test_binary = y_test.map({"Legitimate": 0, "Phishing": 1}).to_numpy()

    models = _build_models()
    for name, model in models.items():
        model.fit(x_train, y_train)
        phishing_scores = _phishing_scores(model, x_test)
        predictions = _labels_from_scores(phishing_scores, DEFAULT_THRESHOLD)

        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "Precision": round(float(precision_score(y_test, predictions, pos_label="Phishing", zero_division=0)), 4),
                "Recall": round(float(recall_score(y_test, predictions, pos_label="Phishing", zero_division=0)), 4),
                "F1-Score": round(float(f1_score(y_test, predictions, pos_label="Phishing", zero_division=0)), 4),
                "ROC-AUC": round(float(roc_auc_score(y_test_binary, phishing_scores)), 4),
                "PR-AUC": round(float(average_precision_score(y_test_binary, phishing_scores)), 4),
                "Brier Score": round(float(brier_score_loss(y_test_binary, phishing_scores)), 4),
            }
        )
        confusion_by_model[name] = confusion_matrix(y_test, predictions, labels=LABELS).tolist()
        threshold_analysis[name] = _threshold_rows(y_test_binary, phishing_scores)
        roc_rows, pr_rows, calibration_rows = _curve_rows(name, y_test_binary, phishing_scores)
        roc_rows_all.extend(roc_rows)
        pr_rows_all.extend(pr_rows)
        calibration_rows_all.extend(calibration_rows)

        if name == FINAL_MODEL_NAME:
            error_analysis = _error_examples(test_df, predictions, phishing_scores)

    final_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    final_features = final_vectorizer.fit_transform(dataset["text"].map(preprocess_text))
    final_models = _build_models()
    for model in final_models.values():
        model.fit(final_features, dataset["label"])

    final_explainers = _build_explainers()
    for explainer in final_explainers.values():
        explainer.fit(final_features, dataset["label"])

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "labels": LABELS,
        "final_model": FINAL_MODEL_NAME,
        "default_threshold": DEFAULT_THRESHOLD,
        "model_names": list(final_models.keys()),
        "dataset_rows": int(len(dataset)),
        "dataset_original_rows": int(dataset.attrs.get("dataset_original_rows", len(dataset))),
        "dataset_source": dataset.attrs.get("dataset_source", "Unknown"),
        "dataset_source_url": dataset.attrs.get("dataset_source_url", ""),
        "dataset_cache_path": dataset.attrs.get("dataset_cache_path", ""),
        "dataset_max_rows": dataset.attrs.get("dataset_max_rows"),
        "dataset_load_warning": dataset.attrs.get("dataset_load_warning", ""),
        "huggingface_dataset_id": HF_DATASET_ID,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "model_comparison": comparison_rows,
        "confusion_matrices": confusion_by_model,
        "threshold_analysis": threshold_analysis,
        "roc_curve": roc_rows_all,
        "precision_recall_curve": pr_rows_all,
        "calibration_curve": calibration_rows_all,
        "cross_validation": _cross_validation_rows(dataset),
        "error_analysis": error_analysis,
        "preprocessing": [
            "case folding",
            "url tokenization",
            "email tokenization",
            "special character removal",
            "stopword removal",
            "lightweight lemmatization",
        ],
        "advanced_features": [
            "calibrated probability models",
            "threshold tuning",
            "ROC and precision-recall curves",
            "5-fold cross-validation",
            "false positive and false negative analysis",
            "static URL risk parsing",
            "phishing pattern categories",
            "local TF-IDF contribution explanation",
        ],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_models[FINAL_MODEL_NAME], MODEL_PATH)
    joblib.dump(final_models, MODEL_COLLECTION_PATH)
    joblib.dump(final_explainers, EXPLAINERS_PATH)
    joblib.dump(final_vectorizer, VECTORIZER_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ModelBundle(
        model=final_models[FINAL_MODEL_NAME],
        vectorizer=final_vectorizer,
        metadata=metadata,
        dataset=dataset,
        models=final_models,
        explainers=final_explainers,
    )


def _metadata_matches_dataset(metadata: dict[str, Any], dataset: pd.DataFrame) -> bool:
    return (
        metadata.get("artifact_version") == ARTIFACT_VERSION
        and metadata.get("dataset_source") == dataset.attrs.get("dataset_source")
        and metadata.get("dataset_rows") == int(len(dataset))
        and metadata.get("dataset_max_rows") == dataset.attrs.get("dataset_max_rows")
    )


def load_or_train() -> ModelBundle:
    dataset = _current_dataset()
    if (
        MODEL_PATH.exists()
        and MODEL_COLLECTION_PATH.exists()
        and EXPLAINERS_PATH.exists()
        and VECTORIZER_PATH.exists()
        and METADATA_PATH.exists()
    ):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if _metadata_matches_dataset(metadata, dataset):
            models = joblib.load(MODEL_COLLECTION_PATH)
            explainers = joblib.load(EXPLAINERS_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return ModelBundle(
                model=models[metadata["final_model"]],
                vectorizer=vectorizer,
                metadata=metadata,
                dataset=dataset,
                models=models,
                explainers=explainers,
            )

    return train_and_save()


def calculate_risk(prediction: str, confidence: float) -> str:
    if prediction != "Phishing":
        return "Low"
    if confidence > 80:
        return "High"
    if 60 <= confidence <= 80:
        return "Medium"
    return "Low-Medium"


def recommendation_for(prediction: str, risk_level: str) -> str:
    if prediction == "Legitimate":
        return "Email terlihat aman, tetapi tetap periksa pengirim dan jangan membagikan password, OTP, atau data sensitif."
    if risk_level == "High":
        return "Jangan klik tautan, jangan balas email, dan laporkan ke tim keamanan atau penyedia layanan terkait."
    if risk_level == "Medium":
        return "Verifikasi pengirim melalui kanal resmi sebelum membuka tautan atau mengirim informasi pribadi."
    return "Periksa ulang isi email, alamat pengirim, dan tautan sebelum melakukan tindakan apa pun."


def _coefficient_vector(explainer: Any) -> tuple[np.ndarray | None, list[Any]]:
    classes = list(getattr(explainer, "classes_", []))
    if hasattr(explainer, "coef_"):
        coef = np.asarray(explainer.coef_[0], dtype=float)
        if classes and classes[0] == "Phishing":
            coef = -coef
        return coef, classes

    if hasattr(explainer, "feature_log_prob_") and classes:
        phishing_index = classes.index("Phishing")
        legitimate_index = classes.index("Legitimate")
        coef = np.asarray(explainer.feature_log_prob_[phishing_index] - explainer.feature_log_prob_[legitimate_index], dtype=float)
        return coef, classes

    return None, classes


def explain_prediction(
    processed_text: str,
    bundle: ModelBundle,
    prediction: str,
    model_name: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    selected_model = model_name or bundle.metadata["final_model"]
    explainer = bundle.explainers.get(selected_model) or bundle.explainers.get(bundle.metadata["final_model"])
    if explainer is None:
        return []

    coef, _ = _coefficient_vector(explainer)
    if coef is None:
        return []

    features = bundle.vectorizer.transform([processed_text])
    if features.nnz == 0:
        return []

    row = features[0]
    feature_names = bundle.vectorizer.get_feature_names_out()
    raw_scores = row.data * coef[row.indices]

    if prediction == "Phishing":
        ranked = sorted(
            ((feature_names[idx], score) for idx, score in zip(row.indices, raw_scores) if score > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        direction = "Mendorong Phishing"
    else:
        ranked = sorted(
            ((feature_names[idx], abs(score)) for idx, score in zip(row.indices, raw_scores) if score < 0),
            key=lambda item: item[1],
            reverse=True,
        )
        direction = "Mendorong Legitimate"

    return [
        {"Token": token, "Impact": round(float(score), 4), "Direction": direction}
        for token, score in ranked[:limit]
    ]


def predict_email(
    text: str,
    bundle: ModelBundle,
    model_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    selected_model_name = model_name or bundle.metadata["final_model"]
    selected_threshold = float(threshold if threshold is not None else bundle.metadata.get("default_threshold", DEFAULT_THRESHOLD))
    model = bundle.models.get(selected_model_name) or bundle.model

    processed = preprocess_text(text)
    features = bundle.vectorizer.transform([processed])
    phishing_probability = float(_phishing_scores(model, features)[0])
    prediction = "Phishing" if phishing_probability >= selected_threshold else "Legitimate"
    confidence = phishing_probability * 100 if prediction == "Phishing" else (1 - phishing_probability) * 100

    risk_level = calculate_risk(prediction, confidence)
    suspicious_keywords = detect_suspicious_keywords(text)
    explanation = explain_prediction(processed, bundle, prediction, selected_model_name)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "phishing_probability": round(phishing_probability * 100, 2),
        "threshold": round(selected_threshold, 2),
        "model_name": selected_model_name,
        "risk_level": risk_level,
        "suspicious_keywords": suspicious_keywords,
        "security_indicators": security_indicators(text),
        "url_analysis": analyze_urls(text),
        "pattern_categories": phishing_pattern_categories(text),
        "explanation": explanation,
        "recommendation": recommendation_for(prediction, risk_level),
        "processed_text": processed,
        "input_stats": {
            "word_count": word_count(text),
            "url_count": count_urls(text),
            "email_count": count_email_addresses(text),
            "suspicious_keyword_count": len(suspicious_keywords),
        },
    }
