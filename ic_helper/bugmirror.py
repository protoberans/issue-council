import os
import re
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from .config import (
    BUGMIRROR_FILE,
    BUGMIRROR_MAX_AGE_DAYS,
    BUGMIRROR_EXCLUDE_STATUSES,
    USE_BIGRAMS,
)
from .text_utils import tokenize, bigrams, TAG_SPLIT_RE

SHIP_TAG_RE = re.compile(r"^[A-Z]{2,6}-[A-Za-z0-9][A-Za-z0-9\-]*$")
TAG_IGNORE_PREFIXES = ("LIVE-", "PTU-", "EPTU-", "TECH-", "ALPHA-", "BETA-", "HOTFIX-")

TRUE_TAIL_RE = re.compile(r"\s+TRUE\s*$", re.IGNORECASE)

DEV_STATUS_PHRASES = [
    "devs investigating",
    "handed off to devs",
    "qa investigating",
    "unable to reproduce",
]

DEV_TAIL_RE = re.compile(
    r"(?:\s+TRUE)?\s+(?:devs investigating|handed off to devs|qa investigating|unable to reproduce)\s*$",
    re.IGNORECASE,
)

def detect_is_main(summary: str) -> bool:
    if not summary:
        return False
    s = summary.strip()
    return bool(re.search(r"\bTRUE\b", s, flags=re.IGNORECASE))

def clean_summary_for_display(summary: str) -> str:
    if not summary:
        return summary

    s = str(summary).strip()
    prev = None
    while prev != s:
        prev = s
        s = DEV_TAIL_RE.sub("", s).strip()
        s = TRUE_TAIL_RE.sub("", s).strip()

    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def classify_dev_status(status: str, summary: str = "") -> Optional[str]:
    s = (status or "").strip().lower()
    for p in DEV_STATUS_PHRASES:
        if p in s:
            return p

    sm = (summary or "").strip().lower()
    for p in DEV_STATUS_PHRASES:
        if p in sm:
            return p

    return None

def ship_token_from_tag(tag: str) -> Optional[str]:
    if not tag or not SHIP_TAG_RE.match(tag):
        return None
    if any(tag.startswith(pfx) for pfx in TAG_IGNORE_PREFIXES):
        return None
    parts = tag.split("-", 1)
    if len(parts) != 2:
        return None
    model = re.sub(r"[^A-Za-z0-9]+", "", parts[1]).lower()
    if len(model) < 3:
        return None
    return model

def is_probable_location_label(tag: str) -> bool:
    if not tag:
        return False
    t = tag.strip()
    if any(t.startswith(pfx) for pfx in TAG_IGNORE_PREFIXES):
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9]{2,}$", t) and not SHIP_TAG_RE.match(t):
        low = t.lower()
        if low in {"cargo", "docking", "impound", "ui", "hud", "audio", "graphics", "mission", "contract"}:
            return False
        return True
    return False

def parse_updated_dt(updated: str) -> Optional[datetime]:
    if not updated:
        return None
    u = updated.strip()
    for fmt in ("%d %B %Y, %H:%M", "%d %b %Y, %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(u, fmt)
        except Exception:
            pass
    return None

@dataclass
class BugmirrorIndex:
    rows: List[Dict[str, Any]]
    df: Dict[str, int]
    df_bg: Dict[str, int]
    n_docs: int
    ship_vocab: Set[str]
    location_vocab: Set[str]
    label_vocab: Set[str]

    @classmethod
    def empty(cls) -> "BugmirrorIndex":
        return cls(rows=[], df={}, df_bg={}, n_docs=0, ship_vocab=set(), location_vocab=set(), label_vocab=set())

    def compute_idf(self, tok: str) -> float:
        d = self.df.get(tok, 0)
        return math.log((self.n_docs + 1) / (d + 1)) + 1.0

    def compute_idf_bigram(self, bg: str) -> float:
        d = self.df_bg.get(bg, 0)
        return math.log((self.n_docs + 1) / (d + 1)) + 1.0

def load_bugmirror(path: str = BUGMIRROR_FILE) -> BugmirrorIndex:
    if not os.path.exists(path):
        print(f"[bugmirror] File not found: {path}")
        return BugmirrorIndex.empty()

    rows: List[Dict[str, Any]] = []
    df: Dict[str, int] = {}
    df_bg: Dict[str, int] = {}
    ship_vocab: Set[str] = set()
    loc_vocab: Set[str] = set()
    label_vocab: Set[str] = set()

    cutoff_dt = None
    if BUGMIRROR_MAX_AGE_DAYS and BUGMIRROR_MAX_AGE_DAYS > 0:
        cutoff_dt = datetime.utcnow() - timedelta(days=BUGMIRROR_MAX_AGE_DAYS)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if not obj.get("issueCode") or not obj.get("issueCouncilUrl"):
                continue

            status_l = (obj.get("status", "") or "").strip().lower()
            if BUGMIRROR_EXCLUDE_STATUSES and status_l in BUGMIRROR_EXCLUDE_STATUSES:
                continue

            updated = obj.get("updated", "") or ""
            dt = parse_updated_dt(updated)
            obj["_updated_dt"] = dt
            if cutoff_dt and dt and dt < cutoff_dt:
                continue

            summary = obj.get("summary", "") or ""
            tags = obj.get("tags", []) or []
            raw = obj.get("raw", "") or ""

            obj["_is_main"] = detect_is_main(summary)
            obj["_summary_clean"] = clean_summary_for_display(summary)
            obj["_dev_status"] = classify_dev_status(obj.get("status", "") or "", summary)

            ships: Set[str] = set()
            locations: Set[str] = set()
            labels: Set[str] = set()

            for tg in tags:
                tg = str(tg).strip()
                if not tg:
                    continue

                st = ship_token_from_tag(tg)
                if st:
                    ships.add(st)
                    ship_vocab.add(st)

                if is_probable_location_label(tg):
                    for lt in tokenize(tg):
                        locations.add(lt)
                        loc_vocab.add(lt)

                if not any(tg.startswith(pfx) for pfx in TAG_IGNORE_PREFIXES):
                    for part in TAG_SPLIT_RE.split(tg):
                        part = part.strip()
                        if not part:
                            continue
                        for t in tokenize(part):
                            labels.add(t)
                            label_vocab.add(t)

            tok_summary_list = tokenize(summary)
            tok_summary = set(tok_summary_list)

            tok_tags: Set[str] = set()
            for tg in tags:
                tok_tags.update(tokenize(str(tg)))

            tok_raw_list = tokenize(raw)
            tok_raw = set(tok_raw_list)

            bgs: Set[str] = set()
            if USE_BIGRAMS:
                bgs.update(bigrams(tok_summary_list))
                if tok_raw_list:
                    bgs.update(bigrams(tok_raw_list[:120]))

            obj["_tok_summary"] = tok_summary
            obj["_tok_tags"] = tok_tags
            obj["_tok_raw"] = tok_raw
            obj["_bg"] = bgs

            obj["_ships"] = ships
            obj["_locations"] = locations
            obj["_labels"] = labels

            doc_tokens = tok_summary | tok_tags | tok_raw
            for t in doc_tokens:
                df[t] = df.get(t, 0) + 1

            if USE_BIGRAMS and bgs:
                for bg in bgs:
                    df_bg[bg] = df_bg.get(bg, 0) + 1

            rows.append(obj)

    idx = BugmirrorIndex(
        rows=rows,
        df=df,
        df_bg=df_bg,
        n_docs=len(rows),
        ship_vocab=ship_vocab,
        location_vocab=loc_vocab,
        label_vocab=label_vocab,
    )
    print(f"[bugmirror] Loaded {idx.n_docs} rows from {path}")
    return idx
