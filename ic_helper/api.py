import os
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    TOP_K,
    CANDIDATE_K,
    PER_TOKEN_CAP,
    openai_is_enabled,
    OPENAI_MODEL,
)
from .models import IssuePayload, MatchResponse
from .features import build_query_features
from .rerank import rerank_candidates
from .text_utils import GENERIC_TOKENS
from .scoring import score_candidate
from .bugmirror import BugmirrorIndex, load_bugmirror


def create_app() -> FastAPI:
    app = FastAPI(title="IC Helper (Local)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://issue-council.robertsspaceindustries.com",
            "http://127.0.0.1",
            "http://localhost",
        ],
        allow_credentials=False,
        allow_methods=["POST", "OPTIONS", "GET"],
        allow_headers=["*"],
    )

    # Single place to keep the in-memory index
    state: Dict[str, Any] = {"idx": load_bugmirror()}

    # ======= STARTUP DIAGNOSTICS (VERBOSE) =======
    print("========== IC HELPER STARTUP ==========")
    print(f"CWD: {os.getcwd()}")
    print(f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"OPENAI enabled (openai_is_enabled): {openai_is_enabled()}")
    print(f"OPENAI_MODEL: {OPENAI_MODEL}")
    print("=======================================")
    # ============================================

    @app.post("/bugmirror/match", response_model=MatchResponse)
    def bugmirror_match(issue: IssuePayload) -> MatchResponse:
        idx: BugmirrorIndex = state["idx"]
        if not idx.rows:
            print("[api] bugmirror_match: idx empty -> returning []")
            return MatchResponse(matches=[])

        print("[api] bugmirror_match: request received")
        print(f"[api] issueCode={issue.issueCode} title={repr(issue.title)[:120]}")

        feats = build_query_features(
            issue,
            ship_vocab=idx.ship_vocab,
            location_vocab=idx.location_vocab,
            label_vocab=idx.label_vocab,
        )
        viewing_issue_code = issue.issueCode or ""

        q_tokens = feats["q_tokens"]
        q_weights = feats["q_weights"]
        q_ships = feats["q_ships"]
        q_scenario = feats["q_scenario"]

        print(
            f"[api] q_tokens={len(q_tokens)} q_ships={sorted(list(q_ships))[:5]} q_scenario={sorted(list(q_scenario))[:5]}"
        )

        candidate_source = idx.rows
        if q_ships:
            filtered = [row for row in idx.rows if row.get("_ships", set()) & q_ships]
            if filtered:
                candidate_source = filtered
                print(f"[api] ship prefilter active -> {len(candidate_source)} candidates")
            else:
                print("[api] ship prefilter found 0 -> using full dataset")

        quick_scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in candidate_source:
            s_tokens = row.get("_tok_summary", set())
            t_tokens = row.get("_tok_tags", set())
            base_set = s_tokens | t_tokens

            overlap = q_tokens & base_set
            if not overlap:
                continue

            sc = 0.0
            scenario_hit = 0
            for tok in overlap:
                wq = q_weights.get(tok, 1.0)
                mult = 0.35 if tok in GENERIC_TOKENS else 1.0
                sc += min(PER_TOKEN_CAP, idx.compute_idf(tok) * 2.0 * wq * mult)
                if tok in q_scenario:
                    scenario_hit += 1

            if scenario_hit:
                sc *= (1.0 + min(0.60, 0.15 * scenario_hit))

            if sc > 0:
                quick_scored.append((sc, row))

        quick_scored.sort(key=lambda x: x[0], reverse=True)
        shortlist = [r for _, r in quick_scored[:CANDIDATE_K]] if quick_scored else []
        print(f"[api] quick shortlist size: {len(shortlist)} (from {len(candidate_source)} candidates)")

        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for row in shortlist:
            sc, explain = score_candidate(
                feats,
                viewing_issue_code,
                row,
                compute_idf=idx.compute_idf,
                compute_idf_bigram=idx.compute_idf_bigram,
            )
            if sc > 0:
                scored.append((sc, row, explain))

        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"[api] fully scored candidates: {len(scored)}")

        local_items: List[Dict[str, Any]] = []
        pool_n = max(TOP_K, 60)  # pool for rerank
        for sc, row, explain in scored[:pool_n]:
            item = {
                "score": round(sc, 2),
                "issueCode": row.get("issueCode"),
                "summary": row.get("_summary_clean") or row.get("summary"),
                "status": row.get("status"),
                "updated": row.get("updated"),
                "tags": (row.get("tags") or [])[:10],
                "issueCouncilUrl": row.get("issueCouncilUrl"),
                "isMain": bool(row.get("_is_main", False)),
                "devStatus": row.get("_dev_status"),
            }
            if explain:
                item["why"] = explain
            local_items.append(item)

        print(f"[api] local_items prepared for return: {len(local_items)}")
        print(f"[api] OpenAI enabled? {openai_is_enabled()}")

        if openai_is_enabled() and local_items:
            print("[api] attempting OpenAI rerank...")
            reranked = rerank_candidates(issue, local_items)
            if reranked:
                print("[api] returning RERANKED results")
                return MatchResponse(matches=reranked[:TOP_K])
            else:
                print("[api] rerank failed / skipped -> returning LOCAL results")

        return MatchResponse(matches=local_items[:TOP_K])

    @app.post("/generate")
    def generate_stub():
        return {"ok": True, "message": "stub"}

    @app.get("/health")
    def health():
        idx: BugmirrorIndex = state["idx"]
        return {
            "ok": True,
            "bugmirror_rows": idx.n_docs,
            "openai": {"enabled": openai_is_enabled(), "model": OPENAI_MODEL},
        }

    @app.post("/reload")
    def reload_bugmirror():
        print("[api] reload requested")
        state["idx"] = load_bugmirror()
        idx: BugmirrorIndex = state["idx"]
        return {"ok": True, "bugmirror_rows": idx.n_docs}

    return app
