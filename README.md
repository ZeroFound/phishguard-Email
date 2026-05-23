# PhishGuard

PhishGuard adalah aplikasi Streamlit untuk mendeteksi email phishing berbasis NLP dan machine learning. Aplikasi mengikuti SRS/PRD pada dokumen `Dokumen_SRS_PRD_PhishGuard.docx`.

Dataset utama terhubung ke Hugging Face:

`https://huggingface.co/datasets/zefang-liu/phishing-email-dataset`

## Fitur

- Input teks email manual.
- Preprocessing: case folding, URL tokenization, email tokenization, special character removal, stopword removal, dan lemmatization ringan.
- Prediksi `Phishing` atau `Legitimate`.
- Confidence score, risk level, suspicious keywords, dan rekomendasi.
- Explainability token TF-IDF yang paling memengaruhi hasil prediksi.
- Security checklist: URL, alamat email, permintaan password/OTP, urgency, ajakan klik link, dan keyword mencurigakan.
- Model calibration dengan calibrated Logistic Regression dan calibrated Linear SVM.
- Threshold tuning untuk mengatur sensitivitas deteksi phishing.
- ROC curve, Precision-Recall curve, calibration curve, ROC-AUC, PR-AUC, dan Brier score.
- 5-fold cross-validation.
- False positive dan false negative analysis.
- Model selector interaktif untuk membandingkan Naive Bayes, calibrated Logistic Regression, dan calibrated Linear SVM saat inference.
- URL risk parser statis tanpa membuka URL.
- Kategori pola phishing: credential theft, financial fraud, urgency pressure, account takeover, malicious link bait, dan scam offer.
- Highlight lokal ala LIME berdasarkan kontribusi token TF-IDF.
- Upload batch file `.txt` atau `.csv`.
- Riwayat prediksi sementara di session state tanpa menyimpan isi email asli.
- Export hasil prediksi, batch, history, metadata, dan evaluasi model ke CSV.
- Halaman Dataset untuk melihat sumber Hugging Face, distribusi label, preview aman, dan konfigurasi sampling.
- Dashboard evaluasi model: accuracy, precision, recall, F1-score, confusion matrix, dan model comparison.
- Halaman informasi project dan kesesuaian requirement.

## Struktur

```text
app.py
data/
  hf_phishing_email_dataset.csv
  sample_emails.csv
model/
  train_model.py
  model.pkl
  models.pkl
  explainers.pkl
  vectorizer.pkl
utils/
  dataset_loader.py
  preprocessing.py
  modeling.py
.streamlit/
  config.toml
requirements.txt
runtime.txt
```

`model.pkl` dan `vectorizer.pkl` dibuat otomatis saat aplikasi pertama kali berjalan, atau dapat dibuat manual dengan:

```bash
python model/train_model.py
```

Secara default aplikasi memakai dataset Hugging Face dan membatasi training ke sampel seimbang 8.000 baris agar ringan untuk Streamlit Community Cloud. Untuk memakai jumlah lain:

```bash
$env:PHISHGUARD_MAX_ROWS="12000"
python model/train_model.py
```

Untuk memakai seluruh dataset, set `PHISHGUARD_MAX_ROWS=0`. Untuk memaksa fallback dataset lokal, set `PHISHGUARD_USE_HF=false`.

## Jalankan Lokal

```bash
pip install -r requirements.txt
python model/train_model.py
streamlit run app.py
```

## Upload Batch

- File `.txt`: dianggap sebagai satu email.
- File `.csv`: pilih kolom teks email, lalu aplikasi menganalisis maksimal 100 baris secara default.
- Hasil batch tidak menampilkan isi email asli. Kolom `Input Hash` dipakai untuk audit ringan tanpa menyimpan data sensitif.

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru, misalnya `phishguard-streamlit`.
2. Upload seluruh isi folder project ini ke repository tersebut.
3. Buka `https://share.streamlit.io`, masuk ke workspace, lalu pilih **Create app**.
4. Pilih **Yup, I have an app**.
5. Isi repository, branch, dan main file path: `app.py`.
6. Buka **Advanced settings** bila ingin memilih versi Python. Versi default Streamlit Community Cloud saat ini adalah Python 3.12.
7. Deploy aplikasi dan pantau log sampai app aktif.

Tidak ada secret yang diperlukan untuk versi demo ini.
