from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_emails.csv"
HF_CACHE_PATH = ROOT_DIR / "data" / "hf_phishing_email_dataset.csv"

HF_DATASET_ID = "zefang-liu/phishing-email-dataset"
HF_DATASET_PAGE = f"https://huggingface.co/datasets/{HF_DATASET_ID}"
HF_CSV_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/Phishing_Email.csv"

LABELS = ["Legitimate", "Phishing"]
HF_LABEL_MAP = {
    "Safe Email": "Legitimate",
    "Phishing Email": "Phishing",
}


def _normalise_project_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"text", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

    normalised = df.dropna(subset=["text", "label"]).copy()
    normalised["text"] = normalised["text"].astype(str)
    normalised["label"] = normalised["label"].astype(str).str.strip().str.title()
    normalised = normalised[normalised["label"].isin(LABELS)]
    return normalised.reset_index(drop=True)


def _normalise_huggingface_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Email Text", "Email Type"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Hugging Face dataset missing columns: {', '.join(sorted(missing))}")

    normalised = df.dropna(subset=["Email Text", "Email Type"]).copy()
    normalised = normalised.rename(columns={"Email Text": "text", "Email Type": "label"})
    normalised["text"] = normalised["text"].astype(str)
    normalised["label"] = normalised["label"].map(HF_LABEL_MAP)
    normalised = normalised.dropna(subset=["label"])
    return normalised[["text", "label"]].reset_index(drop=True)


def _balanced_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)

    per_label = max(1, max_rows // len(LABELS))
    sampled_parts = [
        group.sample(min(len(group), per_label), random_state=42)
        for _, group in df.groupby("label", group_keys=False)
    ]
    sampled = pd.concat(sampled_parts)

    if len(sampled) < max_rows:
        remaining = df.drop(sampled.index, errors="ignore")
        extra = remaining.sample(min(len(remaining), max_rows - len(sampled)), random_state=42)
        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=42).reset_index(drop=True)


def _max_hf_rows() -> int:
    raw_value = os.getenv("PHISHGUARD_MAX_ROWS", "8000")
    try:
        return int(raw_value)
    except ValueError:
        return 8000


def load_sample_dataset() -> pd.DataFrame:
    df = _normalise_project_dataset(pd.read_csv(SAMPLE_DATA_PATH))
    df.attrs.update(
        {
            "dataset_source": "Local demo dataset",
            "dataset_source_url": str(SAMPLE_DATA_PATH),
            "dataset_original_rows": len(df),
            "dataset_cache_path": str(SAMPLE_DATA_PATH),
        }
    )
    return df


def load_huggingface_dataset(force_refresh: bool = False) -> pd.DataFrame:
    if HF_CACHE_PATH.exists() and not force_refresh:
        df = _normalise_project_dataset(pd.read_csv(HF_CACHE_PATH))
        original_rows = len(df)
    else:
        raw_df = pd.read_csv(
            HF_CSV_URL,
            usecols=["Email Text", "Email Type"],
            on_bad_lines="skip",
            encoding_errors="ignore",
        )
        original_rows = len(raw_df)
        df = _normalise_huggingface_dataset(raw_df)
        HF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(HF_CACHE_PATH, index=False)

    max_rows = _max_hf_rows()
    df = _balanced_sample(df, max_rows)
    df.attrs.update(
        {
            "dataset_source": f"Hugging Face: {HF_DATASET_ID}",
            "dataset_source_url": HF_DATASET_PAGE,
            "dataset_original_rows": original_rows,
            "dataset_cache_path": str(HF_CACHE_PATH),
            "dataset_max_rows": max_rows,
        }
    )
    return df


def load_dataset(prefer_huggingface: bool = True, force_refresh: bool = False) -> pd.DataFrame:
    if prefer_huggingface:
        try:
            return load_huggingface_dataset(force_refresh=force_refresh)
        except Exception as exc:
            fallback = load_sample_dataset()
            fallback.attrs["dataset_load_warning"] = str(exc)
            return fallback

    return load_sample_dataset()
