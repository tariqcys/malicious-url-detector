import re
import numpy as np
import tldextract

SUSPICIOUS_WORDS = ["login","verify","update","secure","account","free","confirm","bank","paypal"]

def has_ip(url):
    return 1 if re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", url) else 0

def extract_features(url):
    ext = tldextract.extract(url)
    domain = ext.domain + "." + ext.suffix

    url_length = len(url)
    domain_length = len(domain)
    num_dots = url.count(".")
    num_dashes = url.count("-")
    num_slashes = url.count("/")
    is_https = 1 if url.startswith("https") else 0
    has_at = 1 if "@" in url else 0
    has_ip_flag = has_ip(url)
    has_query = 1 if "?" in url else 0
    suspicious_count = sum(1 for w in SUSPICIOUS_WORDS if w in url.lower())
    tld_len = len(ext.suffix)

    return np.array([
        url_length,
        domain_length,
        num_dots,
        num_dashes,
        num_slashes,
        is_https,
        has_at,
        has_ip_flag,
        has_query,
        suspicious_count,
        tld_len,
        len(ext.domain)
    ])
