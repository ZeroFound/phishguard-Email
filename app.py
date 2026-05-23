from __future__ import annotations

import html
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.modeling import LABELS, load_or_train, predict_email, train_and_save
from utils.preprocessing import (
    count_email_addresses,
    count_urls,
    detect_suspicious_keywords,
    top_tokens,
    word_count,
)


st.set_page_config(
    page_title="PhishGuard",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Menyiapkan model machine learning...")
def get_bundle():
    return load_or_train()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --pg-ink: #17202a;
            --pg-muted: #5f6b7a;
            --pg-line: #d9e0e8;
            --pg-soft: #f6f8fb;
            --pg-teal: #007c89;
            --pg-red: #c73535;
            --pg-amber: #a86600;
            --pg-green: #237a3b;
            --pg-blue: #2b5dab;
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--pg-ink);
        }
        .pg-subtitle {
            color: var(--pg-muted);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 920px;
        }
        .pg-hero {
            border: 1px solid var(--pg-line);
            border-radius: 8px;
            background: #ffffff;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
        }
        .pg-chip {
            display: inline-block;
            border: 1px solid var(--pg-line);
            background: var(--pg-soft);
            color: var(--pg-ink);
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            font-size: 0.88rem;
            white-space: nowrap;
        }
        .pg-chip-red {
            border-color: #efc2c2;
            background: #fff3f3;
            color: var(--pg-red);
        }
        .pg-chip-green {
            border-color: #bfe2c8;
            background: #f1fbf4;
            color: var(--pg-green);
        }
        .pg-small {
            color: var(--pg-muted);
            font-size: 0.9rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--pg-line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: #ffffff;
        }
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }
        .pg-highlight {
            border: 1px solid var(--pg-line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.9rem 1rem;
            line-height: 1.7;
            max-height: 260px;
            overflow: auto;
            white-space: pre-wrap;
        }
        .pg-mark-red {
            background: #ffe0e0;
            color: #8a1f1f;
            border-radius: 4px;
            padding: 0.05rem 0.2rem;
        }
        .pg-mark-green {
            background: #ddf5e5;
            color: #1f6d36;
            border-radius: 4px;
            padding: 0.05rem 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("email_input", "")
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("prediction_history", [])
    st.session_state.setdefault("batch_results", None)


def set_example() -> None:
    st.session_state.email_input = (
        "Dear user, your account has been suspended. Please verify your account "
        "immediately by clicking the link below: https://account-review.example/login"
    )


def reset_input() -> None:
    st.session_state.email_input = ""
    st.session_state.last_result = None


def short_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _risk_score_rank(level: str) -> int:
    return {"Low": 1, "Medium": 2, "High": 3}.get(level, 0)


def _max_url_risk(result: dict[str, Any]) -> str:
    rows = result.get("url_analysis", [])
    if not rows:
        return "None"
    return max((str(row["Risk Level"]) for row in rows), key=_risk_score_rank)


def _pattern_summary(result: dict[str, Any]) -> str:
    categories = [
        str(row["Category"])
        for row in result.get("pattern_categories", [])
        if row.get("Risk") != "Low"
    ]
    return ", ".join(categories[:5]) if categories else "None"


def highlighted_email_html(text: str, explanation: list[dict[str, Any]]) -> str:
    phishing_terms = set()
    legitimate_terms = set()
    for item in explanation:
        target = phishing_terms if item.get("Direction") == "Mendorong Phishing" else legitimate_terms
        for token in str(item.get("Token", "")).split():
            if len(token) >= 3:
                target.add(token.lower())

    if not phishing_terms and not legitimate_terms:
        return f"<div class='pg-highlight'>{html.escape(text)}</div>"

    pieces = []
    for piece in re.findall(r"\w+|\W+", text or ""):
        lowered = piece.lower()
        escaped = html.escape(piece)
        if lowered in phishing_terms:
            pieces.append(f"<mark class='pg-mark-red'>{escaped}</mark>")
        elif lowered in legitimate_terms:
            pieces.append(f"<mark class='pg-mark-green'>{escaped}</mark>")
        else:
            pieces.append(escaped)

    return f"<div class='pg-highlight'>{''.join(pieces)}</div>"


def result_to_record(result: dict[str, Any], text: str, source: str, row_number: int | None = None) -> dict[str, Any]:
    stats = result["input_stats"]
    return {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Row": row_number if row_number is not None else "",
        "Input Hash": short_hash(text),
        "Model": result.get("model_name", ""),
        "Threshold": result.get("threshold", ""),
        "Prediction": result["prediction"],
        "Confidence": result["confidence"],
        "Phishing Probability": result.get("phishing_probability", ""),
        "Risk Level": result["risk_level"],
        "Max URL Risk": _max_url_risk(result),
        "Pattern Categories": _pattern_summary(result),
        "Words": stats["word_count"],
        "URLs": stats["url_count"],
        "Email Addresses": stats["email_count"],
        "Suspicious Keyword Count": stats["suspicious_keyword_count"],
        "Suspicious Keywords": ", ".join(result["suspicious_keywords"]),
        "Top Explanation": ", ".join(item["Token"] for item in result["explanation"][:5]),
        "Recommendation": result["recommendation"],
    }


def add_history_record(result: dict[str, Any], text: str, source: str, row_number: int | None = None) -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    record = result_to_record(result, text, source, row_number)
    st.session_state.prediction_history.insert(0, record)
    st.session_state.prediction_history = st.session_state.prediction_history[:100]


def render_keyword_chips(keywords: list[str]) -> None:
    if not keywords:
        st.info("Tidak ditemukan kata mencurigakan dominan.")
        return

    chips = "".join(f"<span class='pg-chip pg-chip-red'>{keyword}</span>" for keyword in keywords)
    st.markdown(chips, unsafe_allow_html=True)


def render_header(bundle) -> None:
    st.markdown(
        """
        <div class="pg-hero">
            <h1 style="margin-bottom: 0.3rem;">PhishGuard</h1>
            <p class="pg-subtitle" style="margin-bottom: 0;">
            Deteksi email phishing berbasis NLP, TF-IDF, dan machine learning. Sistem menampilkan
            prediksi, confidence score, risk level, indikator keamanan, dan alasan fitur yang memengaruhi model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    overview_cols = st.columns(4)
    overview_cols[0].metric("Dataset", f"{bundle.metadata.get('dataset_rows', 0):,} baris")
    overview_cols[1].metric("Model Final", bundle.metadata.get("final_model", "Unknown"))
    overview_cols[2].metric("Sumber", "Hugging Face" if "Hugging Face" in bundle.metadata.get("dataset_source", "") else "Lokal")
    overview_cols[3].metric("Mode", "Private Session")


def render_result_panel(
    result: dict[str, Any],
    export_name: str = "phishguard_result.csv",
    source_text: str | None = None,
) -> None:
    status_color = "pg-chip-red" if result["prediction"] == "Phishing" else "pg-chip-green"
    st.subheader("Hasil Prediksi")
    st.markdown(
        f"<span class='pg-chip {status_color}'>{result['prediction']}</span>"
        f"<span class='pg-chip'>Confidence {result['confidence']:.2f}%</span>"
        f"<span class='pg-chip'>Phishing Probability {result.get('phishing_probability', 0):.2f}%</span>"
        f"<span class='pg-chip'>Threshold {result.get('threshold', 0.5):.2f}</span>"
        f"<span class='pg-chip'>Risk {result['risk_level']}</span>",
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Prediksi", result["prediction"])
    metric_cols[1].metric("Phishing Prob.", f"{result.get('phishing_probability', 0):.2f}%")
    metric_cols[2].metric("Confidence", f"{result['confidence']:.2f}%")
    metric_cols[3].metric("Risk Level", result["risk_level"])
    metric_cols[4].metric("Keyword", result["input_stats"]["suspicious_keyword_count"])

    detail_tabs = st.tabs(
        [
            "Rekomendasi",
            "Security Checklist",
            "Explainability",
            "URL Risk",
            "Pattern Categories",
            "Preprocessing",
            "Export",
        ]
    )
    with detail_tabs[0]:
        st.write(result["recommendation"])
        st.markdown("**Suspicious Keywords**")
        render_keyword_chips(result["suspicious_keywords"])
        st.caption(f"Model: {result.get('model_name', '-')}")

    with detail_tabs[1]:
        indicator_df = pd.DataFrame(result["security_indicators"])
        st.dataframe(indicator_df, use_container_width=True, hide_index=True)

    with detail_tabs[2]:
        explanation_df = pd.DataFrame(result["explanation"])
        if explanation_df.empty:
            st.info("Tidak ada token dominan yang dapat dijelaskan dari input ini.")
        else:
            st.bar_chart(explanation_df.set_index("Token")["Impact"])
            st.dataframe(explanation_df, use_container_width=True, hide_index=True)
            if source_text:
                st.markdown("**Highlight Lokal**")
                st.markdown(highlighted_email_html(source_text, result["explanation"]), unsafe_allow_html=True)

    with detail_tabs[3]:
        url_df = pd.DataFrame(result.get("url_analysis", []))
        if url_df.empty:
            st.info("Tidak ada URL yang ditemukan.")
        else:
            st.dataframe(url_df, use_container_width=True, hide_index=True)

    with detail_tabs[4]:
        pattern_df = pd.DataFrame(result.get("pattern_categories", []))
        st.dataframe(pattern_df, use_container_width=True, hide_index=True)

    with detail_tabs[5]:
        stat_cols = st.columns(4)
        stat_cols[0].metric("Jumlah Kata", result["input_stats"]["word_count"])
        stat_cols[1].metric("URL", result["input_stats"]["url_count"])
        stat_cols[2].metric("Alamat Email", result["input_stats"]["email_count"])
        stat_cols[3].metric("Keyword", result["input_stats"]["suspicious_keyword_count"])
        st.code(result["processed_text"] or "(kosong setelah preprocessing)", language="text")

    with detail_tabs[6]:
        export_df = pd.DataFrame(
            [
                {
                    "Model": result.get("model_name", ""),
                    "Threshold": result.get("threshold", ""),
                    "Prediction": result["prediction"],
                    "Confidence": result["confidence"],
                    "Phishing Probability": result.get("phishing_probability", ""),
                    "Risk Level": result["risk_level"],
                    "Max URL Risk": _max_url_risk(result),
                    "Pattern Categories": _pattern_summary(result),
                    "Suspicious Keywords": ", ".join(result["suspicious_keywords"]),
                    "Top Explanation": ", ".join(item["Token"] for item in result["explanation"][:5]),
                    "Recommendation": result["recommendation"],
                }
            ]
        )
        st.download_button(
            "Download hasil CSV",
            data=dataframe_to_csv_bytes(export_df),
            file_name=export_name,
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )


def render_manual_input(bundle, model_name: str, threshold: float) -> None:
    col_input, col_context = st.columns([1.55, 1], gap="large")

    with col_input:
        st.subheader("Input Manual")
        email_text = st.text_area(
            "Masukkan isi email",
            key="email_input",
            height=280,
            placeholder="Tempel isi email yang ingin dianalisis...",
        )

        action_col, reset_col, example_col = st.columns([1, 1, 1.25])
        detect_clicked = action_col.button(
            "Deteksi Email",
            type="primary",
            icon=":material/search:",
            use_container_width=True,
        )
        reset_col.button(
            "Reset",
            icon=":material/refresh:",
            use_container_width=True,
            on_click=reset_input,
        )
        example_col.button(
            "Contoh Phishing",
            icon=":material/content_paste:",
            use_container_width=True,
            on_click=set_example,
        )

        if detect_clicked:
            clean_text = email_text.strip()
            if not clean_text:
                st.warning("Input tidak boleh kosong.")
            elif word_count(clean_text) < 6:
                st.warning("Input terlalu pendek. Masukkan isi email yang lebih lengkap.")
            else:
                result = predict_email(clean_text, bundle, model_name=model_name, threshold=threshold)
                st.session_state.last_result = result
                add_history_record(result, clean_text, "Manual")

    with col_context:
        st.subheader("Pemeriksaan Cepat")
        live_text = st.session_state.email_input
        quick_cols = st.columns(2)
        quick_cols[0].metric("Kata", word_count(live_text))
        quick_cols[1].metric("URL", count_urls(live_text))
        quick_cols[0].metric("Alamat Email", count_email_addresses(live_text))
        quick_cols[1].metric("Keyword", len(detect_suspicious_keywords(live_text)))

        st.info(
            "Input tidak dieksekusi, URL tidak dibuka, dan aplikasi tidak menyimpan isi email asli. "
            "Riwayat sesi hanya menyimpan metadata hasil prediksi."
        )

    if st.session_state.last_result:
        st.divider()
        render_result_panel(st.session_state.last_result, source_text=st.session_state.email_input)


def _find_default_text_column(df: pd.DataFrame) -> int:
    preferred = ["Email Text", "text", "email", "body", "message", "content"]
    lower_columns = {str(column).lower(): index for index, column in enumerate(df.columns)}
    for column in preferred:
        if column.lower() in lower_columns:
            return lower_columns[column.lower()]
    return 0


def _read_uploaded_text(uploaded_file) -> str:
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def _predict_batch(
    text_items: list[tuple[int, str]],
    bundle,
    source: str,
    model_name: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0, text="Menganalisis batch...")
    total = max(1, len(text_items))

    for position, (row_number, text) in enumerate(text_items, start=1):
        clean_text = text.strip()
        if word_count(clean_text) < 3:
            rows.append(
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Source": source,
                    "Row": row_number,
                    "Input Hash": short_hash(clean_text),
                    "Model": model_name,
                    "Threshold": threshold,
                    "Prediction": "Skipped",
                    "Confidence": 0.0,
                    "Phishing Probability": 0.0,
                    "Risk Level": "Too Short",
                    "Max URL Risk": "None",
                    "Pattern Categories": "None",
                    "Words": word_count(clean_text),
                    "URLs": count_urls(clean_text),
                    "Email Addresses": count_email_addresses(clean_text),
                    "Suspicious Keyword Count": len(detect_suspicious_keywords(clean_text)),
                    "Suspicious Keywords": "",
                    "Top Explanation": "",
                    "Recommendation": "Input terlalu pendek untuk dianalisis.",
                }
            )
        else:
            result = predict_email(clean_text, bundle, model_name=model_name, threshold=threshold)
            rows.append(result_to_record(result, clean_text, source, row_number))
            add_history_record(result, clean_text, source, row_number)

        progress.progress(position / total, text=f"Menganalisis batch {position}/{total}")

    progress.empty()
    return pd.DataFrame(rows)


def render_batch_upload(bundle, model_name: str, threshold: float) -> None:
    st.subheader("Batch Upload")
    st.markdown(
        "<p class='pg-subtitle'>Upload file TXT untuk satu email atau CSV untuk banyak email. "
        "Hasil batch tidak menampilkan isi email asli, hanya hash dan metadata prediksi.</p>",
        unsafe_allow_html=True,
    )

    upload_col, option_col = st.columns([1.3, 1], gap="large")
    with upload_col:
        uploaded_file = st.file_uploader("Pilih file TXT atau CSV", type=["txt", "csv"])

    with option_col:
        max_rows = st.number_input("Batas baris CSV", min_value=1, max_value=500, value=100, step=10)

    if uploaded_file is None:
        st.info("Upload file untuk mulai analisis batch.")
        return

    filename = uploaded_file.name.lower()
    if filename.endswith(".txt"):
        text = _read_uploaded_text(uploaded_file)
        st.caption(f"File TXT terdeteksi: {word_count(text)} kata, {count_urls(text)} URL.")
        if st.button("Analisis File TXT", type="primary", icon=":material/search:", use_container_width=True):
            result = predict_email(text, bundle, model_name=model_name, threshold=threshold)
            st.session_state.last_result = result
            add_history_record(result, text, "TXT Upload")
            st.session_state.batch_results = pd.DataFrame([result_to_record(result, text, "TXT Upload", 1)])
            render_result_panel(result, export_name="phishguard_txt_result.csv", source_text=text)
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"CSV tidak bisa dibaca: {exc}")
        return

    if df.empty:
        st.warning("CSV kosong.")
        return

    text_column = st.selectbox(
        "Pilih kolom teks email",
        options=list(df.columns),
        index=_find_default_text_column(df),
    )
    st.caption(f"CSV memiliki {len(df)} baris. Aplikasi akan menganalisis maksimal {max_rows} baris.")

    preview_df = df[[text_column]].head(5).copy()
    preview_df[text_column] = preview_df[text_column].astype(str).str.slice(0, 180)
    st.dataframe(preview_df, use_container_width=True)

    if st.button("Analisis CSV", type="primary", icon=":material/table_view:", use_container_width=True):
        selected_texts = [
            (int(index) + 1, str(value))
            for index, value in df[text_column].dropna().head(int(max_rows)).items()
        ]
        batch_df = _predict_batch(selected_texts, bundle, "CSV Upload", model_name, threshold)
        st.session_state.batch_results = batch_df

    if isinstance(st.session_state.batch_results, pd.DataFrame) and not st.session_state.batch_results.empty:
        st.subheader("Hasil Batch")
        batch_df = st.session_state.batch_results
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download hasil batch CSV",
            data=dataframe_to_csv_bytes(batch_df),
            file_name="phishguard_batch_results.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )


def render_history() -> None:
    st.subheader("Riwayat Prediksi Sesi")
    history = st.session_state.prediction_history
    if not history:
        st.info("Belum ada riwayat prediksi pada sesi ini.")
        return

    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download riwayat CSV",
        data=dataframe_to_csv_bytes(history_df),
        file_name="phishguard_session_history.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
    )

    if st.button("Hapus riwayat sesi", icon=":material/delete:", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()


def render_detection(bundle, model_name: str, threshold: float) -> None:
    render_header(bundle)
    input_tab, batch_tab, history_tab = st.tabs(["Deteksi Manual", "Upload Batch", "Riwayat"])
    with input_tab:
        render_manual_input(bundle, model_name, threshold)
    with batch_tab:
        render_batch_upload(bundle, model_name, threshold)
    with history_tab:
        render_history()


def _model_comparison_df(bundle) -> pd.DataFrame:
    df = pd.DataFrame(bundle.metadata["model_comparison"])
    return df.sort_values(["F1-Score", "Accuracy"], ascending=False).reset_index(drop=True)


def render_dashboard(bundle) -> None:
    st.title("Dashboard Evaluasi Model")
    st.markdown(
        "<p class='pg-subtitle'>Evaluasi model mencakup accuracy, precision, recall, F1-score, "
        "confusion matrix, perbandingan model, dan EDA dataset.</p>",
        unsafe_allow_html=True,
    )

    comparison = _model_comparison_df(bundle)
    final_row = comparison[comparison["Model"] == bundle.metadata["final_model"]].iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{final_row['Accuracy']:.2%}")
    metric_cols[1].metric("Precision", f"{final_row['Precision']:.2%}")
    metric_cols[2].metric("Recall", f"{final_row['Recall']:.2%}")
    metric_cols[3].metric("F1-Score", f"{final_row['F1-Score']:.2%}")

    source_cols = st.columns([1.2, 1, 1])
    source_cols[0].caption(f"Sumber dataset: {bundle.metadata.get('dataset_source', 'Unknown')}")
    source_cols[1].caption(f"Baris dipakai: {bundle.metadata.get('dataset_rows', 0):,}")
    source_cols[2].caption(f"Baris asli: {bundle.metadata.get('dataset_original_rows', 0):,}")
    if bundle.metadata.get("dataset_load_warning"):
        st.warning(
            "Dataset Hugging Face tidak bisa diakses saat startup, jadi aplikasi memakai dataset demo lokal. "
            f"Detail: {bundle.metadata['dataset_load_warning']}"
        )

    eval_tab, curve_tab, threshold_tab, error_tab, cv_tab, eda_tab, export_tab = st.tabs(
        [
            "Evaluasi Model",
            "ROC/PR/Calibration",
            "Threshold Tuning",
            "FP/FN Analysis",
            "Cross Validation",
            "EDA Dataset",
            "Export",
        ]
    )

    with eval_tab:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.subheader("Model Comparison")
            display_df = comparison.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]:
                display_df[col] = display_df[col].map(lambda value: f"{value:.2%}")
            display_df["Brier Score"] = display_df["Brier Score"].map(lambda value: f"{value:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with right:
            st.subheader("Confusion Matrix")
            matrix = bundle.metadata["confusion_matrices"][bundle.metadata["final_model"]]
            matrix_df = pd.DataFrame(
                matrix,
                index=[f"Actual {label}" for label in LABELS],
                columns=[f"Predicted {label}" for label in LABELS],
            )
            st.dataframe(matrix_df, use_container_width=True)
            st.caption("Model final memakai calibrated Logistic Regression agar confidence lebih stabil.")

    with curve_tab:
        curve_cols = st.columns(2, gap="large")
        roc_df = pd.DataFrame(bundle.metadata.get("roc_curve", []))
        pr_df = pd.DataFrame(bundle.metadata.get("precision_recall_curve", []))
        calibration_df = pd.DataFrame(bundle.metadata.get("calibration_curve", []))

        with curve_cols[0]:
            st.subheader("ROC Curve")
            if roc_df.empty:
                st.info("Data ROC belum tersedia.")
            else:
                st.line_chart(roc_df, x="FPR", y="TPR", color="Model")

        with curve_cols[1]:
            st.subheader("Precision-Recall Curve")
            if pr_df.empty:
                st.info("Data Precision-Recall belum tersedia.")
            else:
                st.line_chart(pr_df, x="Recall", y="Precision", color="Model")

        st.subheader("Calibration Curve")
        if calibration_df.empty:
            st.info("Data calibration curve belum tersedia.")
        else:
            st.line_chart(
                calibration_df,
                x="Mean Predicted Probability",
                y="Observed Phishing Rate",
                color="Model",
            )
            st.caption("Semakin dekat kurva ke garis diagonal ideal, semakin baik kalibrasi probabilitas.")

    with threshold_tab:
        model_names = bundle.metadata.get("model_names", [bundle.metadata["final_model"]])
        selected_model = st.selectbox("Model untuk threshold tuning", options=model_names)
        threshold_df = pd.DataFrame(bundle.metadata.get("threshold_analysis", {}).get(selected_model, []))
        if threshold_df.empty:
            st.info("Data threshold belum tersedia.")
        else:
            st.line_chart(threshold_df, x="Threshold", y=["Precision", "Recall", "F1-Score"])
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)
            st.info(
                "Threshold rendah meningkatkan recall phishing, sedangkan threshold tinggi biasanya "
                "mengurangi false positive."
            )

    with error_tab:
        errors_df = pd.DataFrame(bundle.metadata.get("error_analysis", []))
        if errors_df.empty:
            st.success("Tidak ada false positive atau false negative pada split evaluasi model final.")
        else:
            err_cols = st.columns(2)
            err_cols[0].metric("False Positive", int((errors_df["Error Type"] == "False Positive").sum()))
            err_cols[1].metric("False Negative", int((errors_df["Error Type"] == "False Negative").sum()))
            st.dataframe(errors_df, use_container_width=True, hide_index=True)

    with cv_tab:
        cv_df = pd.DataFrame(bundle.metadata.get("cross_validation", []))
        if cv_df.empty:
            st.info("Data cross-validation belum tersedia.")
        else:
            display_cv = cv_df.copy()
            for col in display_cv.columns:
                if col != "Model":
                    display_cv[col] = display_cv[col].map(lambda value: f"{value:.4f}")
            st.dataframe(display_cv, use_container_width=True, hide_index=True)

    with eda_tab:
        dataset = bundle.dataset.copy()
        dataset["word_count"] = dataset["text"].map(word_count)
        dataset["char_count"] = dataset["text"].str.len()
        dataset["url_count"] = dataset["text"].map(count_urls)
        dataset["email_count"] = dataset["text"].map(count_email_addresses)
        dataset["suspicious_count"] = dataset["text"].map(lambda text: len(detect_suspicious_keywords(text)))

        st.subheader("Ringkasan Dataset")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Jumlah Data", f"{len(dataset):,}")
        summary_cols[1].metric("Phishing", f"{(dataset['label'] == 'Phishing').sum():,}")
        summary_cols[2].metric("Legitimate", f"{(dataset['label'] == 'Legitimate').sum():,}")
        summary_cols[3].metric("Rata-rata Kata", f"{dataset['word_count'].mean():.1f}")

        chart_cols = st.columns(2, gap="large")
        with chart_cols[0]:
            st.markdown("**1. Distribusi Label**")
            st.bar_chart(dataset["label"].value_counts())

            st.markdown("**2. Rata-rata Panjang Email**")
            st.bar_chart(dataset.groupby("label")["char_count"].mean())

            st.markdown("**3. Rata-rata Jumlah Kata**")
            st.bar_chart(dataset.groupby("label")["word_count"].mean())

        with chart_cols[1]:
            st.markdown("**4. Keberadaan URL**")
            st.bar_chart(dataset.groupby("label")["url_count"].sum())

            st.markdown("**5. Suspicious Keyword per Label**")
            st.bar_chart(dataset.groupby("label")["suspicious_count"].mean())

            st.markdown("**6. Top Words Phishing vs Legitimate**")
            phishing_words = pd.DataFrame(top_tokens(dataset[dataset["label"] == "Phishing"]["text"]), columns=["Token", "Count"])
            legitimate_words = pd.DataFrame(top_tokens(dataset[dataset["label"] == "Legitimate"]["text"]), columns=["Token", "Count"])
            words_df = phishing_words.merge(legitimate_words, on="Token", how="outer", suffixes=("_Phishing", "_Legitimate")).fillna(0)
            st.dataframe(words_df, use_container_width=True, hide_index=True)

    with export_tab:
        st.download_button(
            "Download model comparison CSV",
            data=dataframe_to_csv_bytes(comparison),
            file_name="phishguard_model_comparison.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )
        metadata_df = pd.DataFrame([bundle.metadata])
        st.download_button(
            "Download metadata model CSV",
            data=dataframe_to_csv_bytes(metadata_df),
            file_name="phishguard_model_metadata.csv",
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
        )


def render_dataset(bundle) -> None:
    st.title("Dataset")
    st.markdown(
        "<p class='pg-subtitle'>Halaman ini menunjukkan sumber dataset, normalisasi label, ukuran data, "
        "dan profil ringkas dataset yang dipakai untuk training.</p>",
        unsafe_allow_html=True,
    )

    metadata = bundle.metadata
    top_cols = st.columns(4)
    top_cols[0].metric("Sumber", "Hugging Face" if "Hugging Face" in metadata.get("dataset_source", "") else "Lokal")
    top_cols[1].metric("Baris Asli", f"{metadata.get('dataset_original_rows', 0):,}")
    top_cols[2].metric("Baris Training", f"{metadata.get('dataset_rows', 0):,}")
    top_cols[3].metric("Batas Sampel", metadata.get("dataset_max_rows") or "Semua")

    link = metadata.get("dataset_source_url")
    if link:
        st.link_button("Buka dataset sumber", link, icon=":material/open_in_new:")

    dataset = bundle.dataset.copy()
    dataset["word_count"] = dataset["text"].map(word_count)
    dataset["url_count"] = dataset["text"].map(count_urls)
    dataset["email_count"] = dataset["text"].map(count_email_addresses)

    profile_tab, preview_tab, config_tab = st.tabs(["Profil", "Preview Aman", "Konfigurasi"])
    with profile_tab:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Distribusi Label")
            st.bar_chart(dataset["label"].value_counts())
            st.dataframe(dataset["label"].value_counts().rename_axis("Label").reset_index(name="Jumlah"), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Profil Teks")
            profile_df = dataset.groupby("label").agg(
                Average_Words=("word_count", "mean"),
                Average_URLs=("url_count", "mean"),
                Average_Email_Addresses=("email_count", "mean"),
            )
            st.dataframe(profile_df.round(2), use_container_width=True)

    with preview_tab:
        st.caption("Preview dipotong agar tidak menampilkan email terlalu panjang.")
        sample_df = dataset.sample(min(20, len(dataset)), random_state=7).copy()
        sample_df["text"] = sample_df["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
        st.dataframe(sample_df[["label", "text", "word_count", "url_count"]], use_container_width=True, hide_index=True)

    with config_tab:
        st.subheader("Mapping Dataset")
        mapping_df = pd.DataFrame(
            [
                {"Sumber Hugging Face": "Email Text", "Dipakai Sebagai": "text"},
                {"Sumber Hugging Face": "Email Type = Safe Email", "Dipakai Sebagai": "Legitimate"},
                {"Sumber Hugging Face": "Email Type = Phishing Email", "Dipakai Sebagai": "Phishing"},
            ]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        st.info(
            "Default training memakai sampel seimbang 8.000 baris. Ubah environment variable "
            "PHISHGUARD_MAX_ROWS untuk memakai jumlah lain, atau 0 untuk seluruh dataset."
        )
        if st.button(
            "Refresh Hugging Face & retrain model",
            type="primary",
            icon=":material/sync:",
            use_container_width=True,
        ):
            with st.spinner("Mengunduh ulang dataset, melatih model, dan menghitung evaluasi advanced..."):
                train_and_save(force_refresh=True)
            get_bundle.clear()
            st.success("Dataset dan model berhasil diperbarui.")
            st.rerun()


def render_about(bundle) -> None:
    st.title("Tentang Project")
    st.markdown(
        "<p class='pg-subtitle'>PhishGuard dibuat sebagai demo UAS untuk klasifikasi email phishing "
        "menggunakan NLP, TF-IDF, dan algoritma machine learning.</p>",
        unsafe_allow_html=True,
    )

    section_cols = st.columns(2, gap="large")
    with section_cols[0]:
        st.subheader("Metode")
        st.markdown(
            """
            - **Dataset utama**: Hugging Face `zefang-liu/phishing-email-dataset`.
            - **Fallback**: `data/sample_emails.csv` dipakai jika koneksi Hugging Face gagal.
            - **Preprocessing**: case folding, token URL/email, hapus karakter khusus, stopword removal, lemmatization ringan.
            - **Feature extraction**: TF-IDF unigram dan bigram dengan batas 5.000 fitur.
            - **Model final**: Logistic Regression agar confidence score dapat ditampilkan langsung.
            - **Model pembanding**: Naive Bayes dan Linear SVM.
            """
        )

        st.subheader("Fitur Tambahan")
        st.markdown(
            """
            - Explainability token TF-IDF yang memengaruhi prediksi.
            - Security checklist untuk URL, email address, kredensial, urgency, dan keyword.
            - Model calibration, threshold tuning, ROC curve, dan precision-recall curve.
            - 5-fold cross-validation dan false positive/false negative analysis.
            - Pemilihan model interaktif: Naive Bayes, calibrated Logistic Regression, dan calibrated Linear SVM.
            - URL risk parser statis tanpa membuka tautan.
            - Kategori pola phishing: credential theft, financial fraud, urgency pressure, account takeover, malicious link bait, dan scam offer.
            - Upload TXT/CSV untuk prediksi batch.
            - Riwayat sesi tanpa menyimpan isi email asli.
            - Export hasil prediksi, batch, history, dan evaluasi model.
            """
        )

    with section_cols[1]:
        st.subheader("Kesesuaian Requirement")
        checklist = pd.DataFrame(
            [
                {"Requirement": "FR-01 Input teks email", "Status": "Tersedia"},
                {"Requirement": "FR-02 Preprocessing input", "Status": "Tersedia"},
                {"Requirement": "FR-03 Prediksi phishing/legitimate", "Status": "Tersedia"},
                {"Requirement": "FR-04 Confidence score", "Status": "Tersedia"},
                {"Requirement": "FR-05 Suspicious keywords", "Status": "Tersedia"},
                {"Requirement": "FR-06 Risk level", "Status": "Tersedia"},
                {"Requirement": "FR-07 Dashboard evaluasi", "Status": "Tersedia"},
                {"Requirement": "FR-08 Informasi project", "Status": "Tersedia"},
                {"Requirement": "Upload batch", "Status": "Tambahan"},
                {"Requirement": "Explainability", "Status": "Tambahan"},
                {"Requirement": "Export CSV", "Status": "Tambahan"},
                {"Requirement": "ROC/PR curve", "Status": "Advanced"},
                {"Requirement": "Cross-validation", "Status": "Advanced"},
                {"Requirement": "Threshold tuning", "Status": "Advanced"},
                {"Requirement": "URL risk parser", "Status": "Advanced"},
            ]
        )
        st.dataframe(checklist, use_container_width=True, hide_index=True)

        st.subheader("Catatan Keamanan")
        st.info(
            "Aplikasi tidak membuka URL dari input, tidak meminta credential asli, dan tidak menyimpan isi email "
            "pada riwayat sesi. Untuk audit, riwayat hanya menyimpan hash pendek dan metadata prediksi."
        )

    with st.expander("Detail Artefak Model"):
        st.json(bundle.metadata)


def main() -> None:
    inject_styles()
    init_state()
    bundle = get_bundle()

    st.sidebar.title("PhishGuard")
    page = st.sidebar.radio(
        "Navigasi",
        ["Deteksi Email", "Dataset", "Dashboard Evaluasi", "Tentang Project"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Dataset: {bundle.metadata['dataset_rows']:,} baris")
    st.sidebar.caption(f"Model final: {bundle.metadata['final_model']}")
    st.sidebar.caption(f"Sumber: {bundle.metadata.get('dataset_source', 'Unknown')}")
    source_url = bundle.metadata.get("dataset_source_url")
    if source_url:
        st.sidebar.link_button("Buka Dataset", source_url, icon=":material/open_in_new:")
    st.sidebar.divider()
    model_names = bundle.metadata.get("model_names", [bundle.metadata["final_model"]])
    default_model = bundle.metadata.get("final_model", model_names[0])
    model_index = model_names.index(default_model) if default_model in model_names else 0
    selected_model = st.sidebar.selectbox("Model prediksi", options=model_names, index=model_index)
    selected_threshold = st.sidebar.slider(
        "Threshold phishing",
        min_value=0.20,
        max_value=0.80,
        value=float(bundle.metadata.get("default_threshold", 0.5)),
        step=0.05,
    )
    st.sidebar.caption("Threshold mengubah batas probabilitas agar email dianggap phishing.")

    if page == "Deteksi Email":
        render_detection(bundle, selected_model, selected_threshold)
    elif page == "Dataset":
        render_dataset(bundle)
    elif page == "Dashboard Evaluasi":
        render_dashboard(bundle)
    else:
        render_about(bundle)


if __name__ == "__main__":
    main()
