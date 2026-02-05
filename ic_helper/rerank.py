import json
import traceback
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_S,
    OPENAI_RERANK_CANDIDATES,
    OPENAI_MAX_OUTPUT_TOKENS,
    TOP_K,
    openai_is_enabled,
)
from .models import IssuePayload

MAX_QUERY_CHARS = 900
MAX_SUMMARY_CHARS = 240
MAX_TAGS = 8
MAX_REASON_CHARS = 160

MAX_OUTPUT_TOKENS = OPENAI_MAX_OUTPUT_TOKENS

_CLIENT = None


def _clip(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _preview(s: str, n: int = 1400) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "\n...[truncated]")


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    from openai import OpenAI
    _CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _CLIENT


def _build_query_text(issue: IssuePayload) -> str:
    parts = []
    if issue.issueCode:
        parts.append(f"IssueCode: {issue.issueCode}")
    if issue.title:
        parts.append(f"Title: {_clip(issue.title, 260)}")
    if issue.reproductionSteps:
        steps = [s for s in issue.reproductionSteps if (s or "").strip()][:5]
        if steps:
            parts.append("Reproduction steps:\n" + "\n".join(f"- {_clip(s, 180)}" for s in steps))
    if issue.whatHappened:
        parts.append("What happened:\n" + _clip(issue.whatHappened, 520))
    if issue.whatShouldHaveHappened:
        parts.append("What should have happened:\n" + _clip(issue.whatShouldHaveHappened, 420))
    if issue.workaround:
        parts.append("Workaround:\n" + _clip(issue.workaround, 280))
    return _clip("\n\n".join(parts).strip(), MAX_QUERY_CHARS)


def _make_payload(issue: IssuePayload, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "query_issue": _build_query_text(issue),
        "top_k": TOP_K,
        "candidates": [
            {
                "issueCode": it.get("issueCode"),
                "summary": _clip(it.get("summary") or "", MAX_SUMMARY_CHARS),
                "status": _clip(it.get("status") or "", 80),
                "tags": (it.get("tags") or [])[:MAX_TAGS],
                "isMain": bool(it.get("isMain", False)),
                "localScore": it.get("score"),
            }
            for it in items
        ],
        "schema": {
            "ranked": [
                {"issueCode": "STARC-123", "score": 0.9, "reason": "short"}
            ]
        },
        "instruction": "Return ranked with up to top_k items (best duplicates). Do not include items not in candidates.",
    }


def _parse_ranked_json(raw: str) -> Optional[List[Dict[str, Any]]]:
    # Try strict JSON first
    try:
        data = json.loads(raw or "{}")
    except Exception:
        return None

    ranked = data.get("ranked", [])
    if not isinstance(ranked, list) or not ranked:
        return None
    return ranked


def openai_rerank(issue: IssuePayload, items: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not openai_is_enabled() or not items:
        return None

    try:
        client = _get_client()

        system = (
            "You rerank duplicate candidates for Star Citizen Issue Council issues using Bugmirror entries. "
            "Rank by SAME underlying bug (same scenario/mechanic), not just same ship/cargo. "
            "Return ONLY valid JSON as a single object."
        )

        payload = _make_payload(issue, items)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            timeout=OPENAI_TIMEOUT_S,
        )

        raw = resp.choices[0].message.content or ""
        ranked = _parse_ranked_json(raw)

        if ranked is None:
            # Print raw preview for debugging
            print("[openai_rerank] JSON parse failed, raw preview:\n" + _preview(raw))

            # One retry: ask the model to ONLY output repaired JSON
            repair_system = "Return ONLY valid JSON. No markdown. No commentary."
            repair_user = (
                "Fix this into valid JSON with the same meaning. "
                "Keep only keys: ranked (list of {issueCode, score, reason}).\n\n"
                "TEXT TO FIX:\n" + raw
            )

            resp2 = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": repair_user},
                ],
                timeout=OPENAI_TIMEOUT_S,
            )

            raw2 = resp2.choices[0].message.content or ""
            ranked = _parse_ranked_json(raw2)
            if ranked is None:
                print("[openai_rerank] repair JSON also failed, raw2 preview:\n" + _preview(raw2))
                return None

        rr: Dict[str, Tuple[float, str]] = {}
        for r in ranked[:TOP_K]:
            code = str(r.get("issueCode") or "").strip()
            if not code:
                continue
            try:
                score = float(r.get("score", 0.0))
            except Exception:
                score = 0.0
            reason = _clip(str(r.get("reason") or "").strip(), MAX_REASON_CHARS)
            rr[code] = (score, reason)

        if not rr:
            return None

        by_code = {str(it.get("issueCode") or "").strip(): it for it in items}

        out: List[Dict[str, Any]] = []
        for r in ranked[:TOP_K]:
            code = str(r.get("issueCode") or "").strip()
            if code in by_code and code in rr:
                it = dict(by_code[code])
                sc, reason = rr[code]
                it["llmScore"] = round(sc, 3)
                it["llmWhy"] = reason
                out.append(it)

        return out if out else None

    except Exception as e:
        print(f"[openai_rerank] failed: {e}")
        traceback.print_exc()
        return None


def rerank_candidates(issue: IssuePayload, local_items: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not openai_is_enabled() or not local_items:
        return None
    rerank_pool = local_items[: min(len(local_items), OPENAI_RERANK_CANDIDATES)]
    return openai_rerank(issue, rerank_pool)
