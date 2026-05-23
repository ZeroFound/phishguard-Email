import re
import ipaddress
from typing import Iterable
from urllib.parse import urlparse

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)

CUSTOM_STOPWORDS = {
    "dear",
    "hello",
    "hi",
    "please",
    "thanks",
    "thank",
    "regards",
    "sincerely",
    "best",
    "user",
    "customer",
    "anda",
    "yang",
    "dan",
    "atau",
    "untuk",
    "dengan",
    "dari",
    "kami",
    "kita",
    "ini",
    "itu",
    "akan",
    "telah",
    "sebagai",
}

STOPWORDS = set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)

SUSPICIOUS_KEYWORDS = [
    "verify",
    "verification",
    "login",
    "password",
    "urgent",
    "account",
    "suspended",
    "click",
    "secure",
    "bank",
    "update",
    "otp",
    "limited",
    "confirm",
    "credential",
    "wallet",
    "invoice",
    "payment",
    "unusual",
    "locked",
    "reset",
    "immediately",
    "expire",
    "hadiah",
    "verifikasi",
    "akun",
    "sandi",
    "segera",
    "klik",
    "rekening",
]

SUSPICIOUS_PHRASES = [
    "account suspended",
    "verify your account",
    "click the link",
    "unusual activity",
    "password reset",
    "limited time",
    "confirm your identity",
    "update your payment",
    "akun diblokir",
    "verifikasi akun",
]

URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "s.id",
    "shorturl.at",
}

PHISHING_PATTERN_RULES = {
    "Credential Theft": [
        "password",
        "otp",
        "credential",
        "login",
        "pin",
        "kata sandi",
        "sandi",
        "kode",
        "verify your account",
    ],
    "Financial Fraud": [
        "bank",
        "rekening",
        "payment",
        "invoice",
        "billing",
        "refund",
        "transfer",
        "wallet",
        "credit card",
    ],
    "Urgency Pressure": [
        "urgent",
        "immediately",
        "final warning",
        "limited time",
        "expire",
        "expires",
        "today",
        "segera",
    ],
    "Account Takeover": [
        "account suspended",
        "locked",
        "suspended",
        "unusual activity",
        "login attempt",
        "akun diblokir",
    ],
    "Malicious Link Bait": [
        "click",
        "klik",
        "link",
        "tautan",
        "secure link",
        "download",
        "view document",
    ],
    "Reward or Scam Offer": [
        "congratulations",
        "gift",
        "reward",
        "winner",
        "claim",
        "hadiah",
        "menang",
        "refund available",
    ],
}


def _simple_lemma(token: str) -> str:
    """Lightweight fallback lemmatizer that avoids external NLTK corpora."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def preprocess_text(text: str) -> str:
    text = (text or "").lower()
    text = URL_PATTERN.sub(" urltoken ", text)
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in text.split():
        if token in {"urltoken", "emailtoken"}:
            tokens.append(token)
            continue
        if token.isdigit() or token in STOPWORDS or len(token) < 2:
            continue
        tokens.append(_simple_lemma(token))

    return " ".join(tokens)


def detect_suspicious_keywords(text: str) -> list[str]:
    normalized = (text or "").lower()
    found = set()

    for phrase in SUSPICIOUS_PHRASES:
        if phrase in normalized:
            found.add(phrase)

    words = set(re.findall(r"[a-zA-Z]+", normalized))
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in words:
            found.add(keyword)

    return sorted(found)


def count_urls(text: str) -> int:
    return len(URL_PATTERN.findall(text or ""))


def count_email_addresses(text: str) -> int:
    return len(EMAIL_PATTERN.findall(text or ""))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in URL_PATTERN.findall(text or ""):
        cleaned = match.strip().rstrip(".,;:!?)]}'\"")
        if cleaned:
            urls.append(cleaned)
    return urls


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def analyze_urls(text: str) -> list[dict[str, str | int]]:
    analyses = []
    for raw_url in extract_urls(text):
        parsed_url = raw_url if re.match(r"^https?://", raw_url, re.IGNORECASE) else f"http://{raw_url}"
        parsed = urlparse(parsed_url)
        hostname = (parsed.hostname or "").lower()
        path_and_query = f"{parsed.path} {parsed.query}".lower()
        flags = []

        if parsed.scheme != "https":
            flags.append("non-https")
        if hostname in URL_SHORTENERS:
            flags.append("url-shortener")
        if _is_ip_address(hostname):
            flags.append("ip-address-domain")
        if hostname.count(".") >= 3:
            flags.append("many-subdomains")
        if "-" in hostname:
            flags.append("hyphenated-domain")
        if "xn--" in hostname:
            flags.append("punycode-domain")
        if len(raw_url) > 120:
            flags.append("very-long-url")
        if re.search(r"(login|verify|update|secure|account|password|bank|otp)", path_and_query):
            flags.append("suspicious-path")
        if re.search(r"(@|%40)", raw_url):
            flags.append("url-obfuscation")

        score = min(100, len(flags) * 18)
        risk_level = "High" if score >= 54 else "Medium" if score >= 18 else "Low"
        analyses.append(
            {
                "URL": raw_url,
                "Domain": hostname or "-",
                "Scheme": parsed.scheme or "-",
                "Risk Score": score,
                "Risk Level": risk_level,
                "Flags": ", ".join(flags) if flags else "none",
            }
        )
    return analyses


def phishing_pattern_categories(text: str) -> list[dict[str, str | int]]:
    normalized = (text or "").lower()
    rows = []
    for category, patterns in PHISHING_PATTERN_RULES.items():
        evidence = [pattern for pattern in patterns if pattern in normalized]
        if evidence:
            score = min(100, 25 + len(evidence) * 15)
            rows.append(
                {
                    "Category": category,
                    "Score": score,
                    "Evidence": ", ".join(evidence[:8]),
                    "Risk": "High" if score >= 70 else "Medium",
                }
            )

    if not rows:
        rows.append(
            {
                "Category": "No dominant phishing pattern",
                "Score": 0,
                "Evidence": "-",
                "Risk": "Low",
            }
        )
    return rows


def security_indicators(text: str) -> list[dict[str, str]]:
    normalized = (text or "").lower()
    keywords = detect_suspicious_keywords(text)
    url_total = count_urls(text)
    email_total = count_email_addresses(text)
    url_analysis = analyze_urls(text)
    high_url_risks = sum(1 for row in url_analysis if row["Risk Level"] == "High")
    has_credential_request = bool(
        re.search(r"\b(password|otp|credential|pin|kata sandi|sandi|kode)\b", normalized)
    )
    has_urgency = bool(
        re.search(r"\b(urgent|immediately|segera|final warning|limited time|expire|expires|today)\b", normalized)
    )
    has_click_request = bool(re.search(r"\b(click|klik|link|tautan)\b", normalized))

    checks = [
        {
            "Indicator": "URL ditemukan",
            "Status": "Ada" if url_total else "Tidak ada",
            "Risk": "High" if high_url_risks else "Medium" if url_total else "Low",
            "Detail": f"{url_total} URL terdeteksi, {high_url_risks} high risk" if url_total else "Tidak ada URL pada teks",
        },
        {
            "Indicator": "Alamat email ditemukan",
            "Status": "Ada" if email_total else "Tidak ada",
            "Risk": "Low" if email_total else "Low",
            "Detail": f"{email_total} alamat email terdeteksi" if email_total else "Tidak ada alamat email",
        },
        {
            "Indicator": "Permintaan password/OTP",
            "Status": "Ada" if has_credential_request else "Tidak ada",
            "Risk": "High" if has_credential_request else "Low",
            "Detail": "Teks meminta kredensial sensitif" if has_credential_request else "Tidak ada permintaan kredensial",
        },
        {
            "Indicator": "Sense of urgency",
            "Status": "Ada" if has_urgency else "Tidak ada",
            "Risk": "Medium" if has_urgency else "Low",
            "Detail": "Ada tekanan waktu atau urgensi" if has_urgency else "Tidak ada tekanan waktu dominan",
        },
        {
            "Indicator": "Ajakan klik tautan",
            "Status": "Ada" if has_click_request else "Tidak ada",
            "Risk": "Medium" if has_click_request else "Low",
            "Detail": "Teks mengarahkan pengguna untuk klik link" if has_click_request else "Tidak ada ajakan klik link",
        },
        {
            "Indicator": "Keyword mencurigakan",
            "Status": "Ada" if keywords else "Tidak ada",
            "Risk": "High" if len(keywords) >= 4 else "Medium" if keywords else "Low",
            "Detail": ", ".join(keywords[:8]) if keywords else "Tidak ditemukan keyword dominan",
        },
    ]
    return checks


def top_tokens(texts: Iterable[str], limit: int = 12) -> list[tuple[str, int]]:
    frequencies: dict[str, int] = {}
    for text in texts:
        for token in preprocess_text(text).split():
            if token in {"urltoken", "emailtoken"}:
                continue
            frequencies[token] = frequencies.get(token, 0) + 1

    return sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:limit]
