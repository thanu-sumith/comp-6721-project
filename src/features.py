"""Hand-crafted numeric features for response-preference baselines.

Every per-response feature is computed for A and B separately; models then see
[A, B, A-B, shared]. Counts are log1p-transformed so differences are log-ratios.
Model names and row ids are never used.
"""
import re
from dataclasses import dataclass

import numpy as np

REFUSAL_RE = re.compile(
    r"(i'?m sorry|i am sorry|i apologi[sz]e|i cannot|i can'?t|i can not|i'?m unable|"
    r"i am unable|i'?m not able|as an ai|as a language model|i don'?t have (?:the ability|access)|"
    r"cannot assist|can'?t assist|not appropriate)", re.I)
LIST_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.M)
HEADER_RE = re.compile(r"^\s*#{1,6}\s+", re.M)
BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
URL_RE = re.compile(r"https?://")
WORD_RE = re.compile(r"\w+")

RESPONSE_FEATURES = [
    "log_chars", "log_words", "log_lines", "log_list_items", "log_headers", "log_bold",
    "log_code_blocks", "log_questions", "log_exclaims", "log_urls", "refusal_hits",
    "is_empty", "avg_word_len", "type_token_ratio", "prompt_jaccard", "prompt_coverage",
]
SHARED_FEATURES = ["log_prompt_chars", "log_prompt_words", "response_jaccard"]


def _words(text):
    return WORD_RE.findall(text.lower())


def response_vector(text, prompt_words):
    words = _words(text)
    wset = set(words)
    n_words = len(words)
    inter = len(wset & prompt_words)
    union = len(wset | prompt_words)
    return [
        np.log1p(len(text)),
        np.log1p(n_words),
        np.log1p(text.count("\n") + (1 if text else 0)),
        np.log1p(len(LIST_RE.findall(text))),
        np.log1p(len(HEADER_RE.findall(text))),
        np.log1p(len(BOLD_RE.findall(text))),
        np.log1p(text.count("```") // 2),
        np.log1p(text.count("?")),
        np.log1p(text.count("!")),
        np.log1p(len(URL_RE.findall(text))),
        float(min(len(REFUSAL_RE.findall(text)), 3)),
        float(len(text.strip()) == 0),
        float(sum(len(w) for w in words) / n_words) if n_words else 0.0,
        float(len(wset) / n_words) if n_words else 0.0,
        float(inter / union) if union else 0.0,
        float(inter / len(prompt_words)) if prompt_words else 0.0,
    ]


def extract(df):
    """Return (FA, FB, shared) float arrays for a parsed dataframe from load_data."""
    fa, fb, sh = [], [], []
    for p, a, b in zip(df.prompt_text, df.response_a_text, df.response_b_text):
        pw = set(_words(p))
        fa.append(response_vector(a, pw))
        fb.append(response_vector(b, pw))
        wa, wb = set(_words(a)), set(_words(b))
        union = len(wa | wb)
        sh.append([np.log1p(len(p)), np.log1p(len(pw)), len(wa & wb) / union if union else 1.0])
    return np.asarray(fa), np.asarray(fb), np.asarray(sh)


@dataclass
class NumericFeatures:
    """Feature container; `design(swap=True)` returns the A/B-flipped view."""
    FA: np.ndarray
    FB: np.ndarray
    shared: np.ndarray

    def __len__(self):
        return len(self.FA)

    def take(self, idx):
        return NumericFeatures(self.FA[idx], self.FB[idx], self.shared[idx])

    def design(self, swap=False):
        a, b = (self.FB, self.FA) if swap else (self.FA, self.FB)
        return np.hstack([a, b, a - b, self.shared])

    def names(self):
        return ([f"A_{n}" for n in RESPONSE_FEATURES] + [f"B_{n}" for n in RESPONSE_FEATURES]
                + [f"diff_{n}" for n in RESPONSE_FEATURES] + SHARED_FEATURES)


@dataclass
class TfidfFeatures:
    """Sparse TF-IDF of response A and response B (shared vocabulary)."""
    XA: object
    XB: object

    def __len__(self):
        return self.XA.shape[0]

    def take(self, idx):
        return TfidfFeatures(self.XA[idx], self.XB[idx])

    def design(self, swap=False):
        from scipy.sparse import hstack
        a, b = (self.XB, self.XA) if swap else (self.XA, self.XB)
        return hstack([a, b, abs(a - b)], format="csr")


def build_tfidf(df, train_mask, max_chars=5000, max_features=50000):
    """Fit the vectoriser on TRAINING responses only, then transform every row."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    ra = df.response_a_text.str.slice(0, max_chars)
    rb = df.response_b_text.str.slice(0, max_chars)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=max_features,
                          sublinear_tf=True, dtype=np.float32)
    mask = np.asarray(train_mask)
    vec.fit(np.concatenate([ra[mask].to_numpy(), rb[mask].to_numpy()]))
    return TfidfFeatures(vec.transform(ra), vec.transform(rb))
