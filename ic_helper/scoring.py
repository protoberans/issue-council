import math
from typing import Any, Dict, List, Set, Tuple

from .config import (
    DEBUG_EXPLAIN,
    USE_BIGRAMS,
    BIGRAM_BOOST,
    BIGRAM_MAX_MATCHES,
    FIELD_MULT,
    PER_TOKEN_CAP,
    MULTIFIELD_BONUS_CAP,
    MULTIFIELD_BONUS_PER_EXTRA,
    SHIP_MATCH_BOOST,
    LOCATION_MATCH_BOOST,
    LABEL_MATCH_BOOST,
    SHIP_MISS_MULT,
    SCENARIO_MISS_MULT,
    GENERIC_ONLY_MULT,
    GENERIC_ONLY_PENALTY,
    LEN_NORM,
    MAIN_MULT,
    STATUS_MULT,
)
from .text_utils import GENERIC_TOKENS

def _length_norm(doc_token_count: int) -> float:
    return 1.0 / (1.0 + 0.06 * math.log(1.0 + doc_token_count))

def score_candidate(
    feats: Dict[str, Any],
    viewing_issue_code: str,
    row: Dict[str, Any],
    *,
    compute_idf,
    compute_idf_bigram,
) -> Tuple[float, Dict[str, Any]]:
    q_tokens: Set[str] = feats["q_tokens"]
    q_weights: Dict[str, float] = feats["q_weights"]
    q_bigrams: Set[str] = feats["q_bigrams"]
    q_ships: Set[str] = feats["q_ships"]
    q_locations: Set[str] = feats["q_locations"]
    q_labels: Set[str] = feats["q_labels"]
    q_scenario: Set[str] = feats["q_scenario"]

    s_tokens: Set[str] = row.get("_tok_summary", set())
    t_tokens: Set[str] = row.get("_tok_tags", set())
    r_tokens: Set[str] = row.get("_tok_raw", set())
    bg: Set[str] = row.get("_bg", set())

    cand_ships: Set[str] = row.get("_ships", set())
    cand_locations: Set[str] = row.get("_locations", set())
    cand_labels: Set[str] = row.get("_labels", set())

    status_raw = (row.get("status", "") or "").strip()
    status = status_raw.lower()
    is_main = bool(row.get("_is_main", False))

    overlap_s = q_tokens & s_tokens
    overlap_t = q_tokens & t_tokens
    overlap_r = q_tokens & r_tokens

    score = 0.0
    contrib_tokens: List[Tuple[float, str, str]] = []
    contrib_bgs: List[Tuple[float, str]] = []
    notes: List[str] = []

    per_tok_best: Dict[str, Tuple[float, str]] = {}
    per_tok_fields: Dict[str, int] = {}

    def consider(tok: str, where: str) -> None:
        wq = q_weights.get(tok, 1.0)
        delta = compute_idf(tok) * FIELD_MULT[where] * wq
        delta = min(delta, PER_TOKEN_CAP)
        cur = per_tok_best.get(tok)
        if cur is None or delta > cur[0]:
            per_tok_best[tok] = (delta, where)
        per_tok_fields[tok] = per_tok_fields.get(tok, 0) + 1

    for tok in overlap_s:
        consider(tok, "summary")
    for tok in overlap_t:
        consider(tok, "tags")
    for tok in overlap_r:
        consider(tok, "raw")

    for tok, (best_delta, where) in per_tok_best.items():
        score += best_delta
        if DEBUG_EXPLAIN:
            contrib_tokens.append((best_delta, tok, where))

    for tok, cnt in per_tok_fields.items():
        if cnt > 1:
            bonus = per_tok_best[tok][0] * min(MULTIFIELD_BONUS_CAP, MULTIFIELD_BONUS_PER_EXTRA * (cnt - 1))
            score += bonus
            if DEBUG_EXPLAIN:
                contrib_tokens.append((bonus, tok, "multifield_bonus"))

    if USE_BIGRAMS and q_bigrams and bg:
        overlap_bg = list(q_bigrams & bg)
        if overlap_bg:
            overlap_bg = overlap_bg[:BIGRAM_MAX_MATCHES]
            for b in overlap_bg:
                delta = compute_idf_bigram(b) * BIGRAM_BOOST
                score += delta
                if DEBUG_EXPLAIN:
                    contrib_bgs.append((delta, b))

    if q_ships:
        shared = q_ships & cand_ships
        if shared:
            score += SHIP_MATCH_BOOST
            notes.append(f"+{SHIP_MATCH_BOOST:g} ship_match({','.join(sorted(list(shared))[:2])})")
        else:
            score *= SHIP_MISS_MULT
            notes.append(f"*{SHIP_MISS_MULT:g} ship_miss")

    if q_locations and cand_locations:
        shared_loc = q_locations & cand_locations
        if shared_loc:
            score += LOCATION_MATCH_BOOST
            notes.append(f"+{LOCATION_MATCH_BOOST:g} location_match({','.join(sorted(list(shared_loc))[:2])})")

    if q_labels and cand_labels:
        shared_lab = (q_labels & cand_labels) - GENERIC_TOKENS
        if shared_lab:
            bump = LABEL_MATCH_BOOST * min(3, len(shared_lab))
            score += bump
            notes.append(f"+{bump:g} label_match({','.join(sorted(list(shared_lab))[:3])})")

    if q_scenario:
        cand_scenario = (s_tokens | t_tokens | r_tokens) & q_scenario
        if not cand_scenario:
            score *= SCENARIO_MISS_MULT
            notes.append(f"*{SCENARIO_MISS_MULT:g} missed_scenario")

    if GENERIC_ONLY_PENALTY:
        overlap_all = overlap_s | overlap_t | overlap_r
        if overlap_all:
            non_generic = [t for t in overlap_all if t not in GENERIC_TOKENS]
            if not non_generic:
                score *= GENERIC_ONLY_MULT
                notes.append(f"*{GENERIC_ONLY_MULT:g} generic_only_overlap")

    if is_main:
        score *= MAIN_MULT
        notes.append(f"*{MAIN_MULT:g} main")

    for key, mult in STATUS_MULT.items():
        if key in status:
            score *= mult
            notes.append(f"*{mult:g} {key}")
            break

    if viewing_issue_code and row.get("issueCode") == viewing_issue_code:
        score *= 0.2
        notes.append("*0.2 self")

    if LEN_NORM:
        doc_len = len(s_tokens | t_tokens | r_tokens)
        ln = _length_norm(doc_len)
        score *= ln
        if DEBUG_EXPLAIN:
            notes.append(f"*{round(ln, 3)} len_norm({doc_len})")

    explain: Dict[str, Any] = {}
    if DEBUG_EXPLAIN:
        contrib_tokens.sort(key=lambda x: x[0], reverse=True)
        contrib_bgs.sort(key=lambda x: x[0], reverse=True)
        explain = {
            "top_token_contrib": [
                {"tok": tok, "where": where, "delta": round(delta, 2)}
                for (delta, tok, where) in contrib_tokens[:12]
            ],
            "top_bigram_contrib": [
                {"bg": bg2, "delta": round(delta, 2)}
                for (delta, bg2) in contrib_bgs[:10]
            ],
            "notes": notes[:18],
        }

    return score, explain
