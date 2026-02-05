from typing import Any, Dict, List, Set

from .config import Q_WEIGHT_CAP, USE_BIGRAMS
from .models import IssuePayload
from .text_utils import tokenize, bigrams, GENERIC_TOKENS, SCENARIO_SIGNALS

def build_query_features(issue: IssuePayload, *, ship_vocab: Set[str], location_vocab: Set[str], label_vocab: Set[str]) -> Dict[str, Any]:
    w_title = 3.0
    w_what = 1.6
    w_should = 1.2
    w_work = 1.1
    w_repro = 1.3

    q_weights: Dict[str, float] = {}
    all_tokens: List[str] = []

    def add(text: str, w: float) -> None:
        nonlocal all_tokens, q_weights
        toks = tokenize(text or "")
        all_tokens.extend(toks)
        for tok in toks:
            q_weights[tok] = min(Q_WEIGHT_CAP, q_weights.get(tok, 0.0) + w)

    add(issue.title or "", w_title)
    add(issue.whatHappened or "", w_what)
    add(issue.whatShouldHaveHappened or "", w_should)
    add(issue.workaround or "", w_work)
    if issue.reproductionSteps:
        add("\n".join(issue.reproductionSteps), w_repro)

    q_tokens = set(all_tokens)
    q_bigrams = set(bigrams(all_tokens)) if USE_BIGRAMS else set()

    q_ships = set([t for t in q_tokens if t in ship_vocab])
    q_locations = set([t for t in q_tokens if t in location_vocab])
    q_labels = set([t for t in q_tokens if t in label_vocab and t not in GENERIC_TOKENS])
    q_scenario = set([t for t in q_tokens if t in SCENARIO_SIGNALS])

    return {
        "q_tokens": q_tokens,
        "q_weights": q_weights,
        "q_bigrams": q_bigrams,
        "q_ships": q_ships,
        "q_locations": q_locations,
        "q_labels": q_labels,
        "q_scenario": q_scenario,
    }
