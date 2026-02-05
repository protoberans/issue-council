import os
import json
from typing import Dict, Set


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None else str(v)


def _env_json(name: str, default_obj):
    """
    Read JSON from env var. If missing/invalid, return default_obj.
    Example:
      STATUS_MULT={"open":1.1,"confirmed":1.08}
      DEV_STATUS_KEYS=["devs investigating","qa investigating"]
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default_obj
    try:
        return json.loads(raw)
    except Exception:
        return default_obj


def _env_csv_set(name: str, default_set: Set[str]) -> Set[str]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default_set
    return {s.strip() for s in str(raw).split(",") if s.strip()}


# =========================================================
# Config (environment-driven)
# =========================================================
LISTEN_HOST = _env_str("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = _env_int("LISTEN_PORT", 8000)
BUGMIRROR_FILE = _env_str("BUGMIRROR_FILE", "bugmirror_structured.jsonl")

# Output / speed knobs
TOP_K = _env_int("TOP_K", 10)
CANDIDATE_K = _env_int("CANDIDATE_K", 800)
DEBUG_EXPLAIN = _env_bool("DEBUG_EXPLAIN", True)

# Optional filtering
BUGMIRROR_MAX_AGE_DAYS = _env_int("BUGMIRROR_MAX_AGE_DAYS", 0)  # 0 = disabled
BUGMIRROR_EXCLUDE_STATUSES = _env_csv_set("BUGMIRROR_EXCLUDE_STATUSES", set())
BUGMIRROR_EXCLUDE_STATUSES = {s.lower() for s in BUGMIRROR_EXCLUDE_STATUSES}

# Query weight cap
Q_WEIGHT_CAP = _env_float("Q_WEIGHT_CAP", 4.0)

# Phrase scoring
USE_BIGRAMS = _env_bool("USE_BIGRAMS", True)
BIGRAM_BOOST = _env_float("BIGRAM_BOOST", 2.2)
BIGRAM_MAX_MATCHES = _env_int("BIGRAM_MAX_MATCHES", 8)

# Field scoring multipliers
FIELD_MULT: Dict[str, float] = {
    "summary": _env_float("MULT_SUMMARY", 2.6),
    "tags": _env_float("MULT_TAGS", 4.2),
    "raw": _env_float("MULT_RAW", 1.8),
}

# Token contribution cap
PER_TOKEN_CAP = _env_float("PER_TOKEN_CAP", 120.0)

# Multifield bonus
MULTIFIELD_BONUS_PER_EXTRA = _env_float("MULTIFIELD_BONUS_PER_EXTRA", 0.09)
MULTIFIELD_BONUS_CAP = _env_float("MULTIFIELD_BONUS_CAP", 0.18)

# Entity boosts
SHIP_MATCH_BOOST = _env_float("SHIP_MATCH_BOOST", 14.0)
LOCATION_MATCH_BOOST = _env_float("LOCATION_MATCH_BOOST", 6.0)
LABEL_MATCH_BOOST = _env_float("LABEL_MATCH_BOOST", 3.0)

SHIP_MISS_MULT = _env_float("SHIP_MISS_MULT", 0.28)
SCENARIO_MISS_MULT = _env_float("SCENARIO_MISS_MULT", 0.55)

GENERIC_ONLY_MULT = _env_float("GENERIC_ONLY_MULT", 0.35)
GENERIC_ONLY_PENALTY = _env_bool("GENERIC_ONLY_PENALTY", True)

LEN_NORM = _env_bool("LEN_NORM", True)

# MAIN boost
MAIN_MULT = _env_float("MAIN_MULT", 1.08)

# Extra status boosts — env-driven via JSON
_DEFAULT_STATUS_MULT = {
    "open": 1.10,
    "confirmed": 1.08,
    "under": 1.06,
    "devs investigating": 1.12,
    "handed off to devs": 1.12,
    "qa investigating": 1.10,
    "unable to reproduce": 1.08,
}
STATUS_MULT = _env_json("STATUS_MULT", _DEFAULT_STATUS_MULT)
if not isinstance(STATUS_MULT, dict):
    STATUS_MULT = _DEFAULT_STATUS_MULT

# Dev status keys — env-driven via JSON array OR comma-separated list
_DEFAULT_DEV_STATUS_KEYS = [
    "devs investigating",
    "handed off to devs",
    "qa investigating",
    "unable to reproduce",
]
_dev_keys = _env_json("DEV_STATUS_KEYS", None)
if _dev_keys is None:
    DEV_STATUS_KEYS = set(_DEFAULT_DEV_STATUS_KEYS)
elif isinstance(_dev_keys, list):
    DEV_STATUS_KEYS = {str(x).strip().lower() for x in _dev_keys if str(x).strip()}
else:
    # fallback: treat as CSV string if someone set DEV_STATUS_KEYS="a,b,c"
    DEV_STATUS_KEYS = {s.strip().lower() for s in str(_dev_keys).split(",") if s.strip()}

# =========================================================
# Optional OpenAI reranker (env-driven)
# =========================================================
OPENAI_API_KEY = _env_str("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = _env_str("OPENAI_MODEL", "gpt-4.1-mini").strip()

OPENAI_RERANK_ENABLED = _env_bool("OPENAI_RERANK_ENABLED", True)
OPENAI_RERANK_CANDIDATES = _env_int("OPENAI_RERANK_CANDIDATES", 60)
OPENAI_TIMEOUT_S = _env_float("OPENAI_TIMEOUT_S", 20.0)
OPENAI_MAX_OUTPUT_TOKENS = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 420)


def openai_is_enabled() -> bool:
    return bool(OPENAI_API_KEY) and OPENAI_RERANK_ENABLED
