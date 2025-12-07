import tldextract
import re
import json
import numpy as np

with open("feature_columns.json", "r") as f:
    FEATURE_COLS = json.load(f)

def extract_features(url: str):
    url = str(url).lower()

    extracted = tldextract.extract(url)
    domain = extracted.domain
    suffix = extracted.suffix
    subdomain = extracted.subdomain

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "num_dots": url.count("."),
        "num_dashes": url.count("-"),
        "is_https": 1 if url.startswith("https") else 0,
        "has_ip": 1 if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", url) else 0,
        "has_login": 1 if "login" in url else 0,
        "has_secure": 1 if "secure" in url else 0,
        "has_update": 1 if "update" in url else 0,
        "has_free": 1 if "free" in url else 0,
        "has_confirm": 1 if "confirm" in url else 0,
        "num_slashes": url.count("/"),
    }

    return np.array([features[col] for col in FEATURE_COLS], dtype=float)
