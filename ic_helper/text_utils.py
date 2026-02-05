import re
from typing import List

TOKEN_RE = re.compile(r"[a-z0-9]+")
TAG_SPLIT_RE = re.compile(r"[,\s/|]+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "without",
    "is", "are", "was", "were", "it", "this", "that", "as", "at", "by", "from", "into",
    "then", "when", "while", "after", "before", "again",
    "player", "game", "issue", "bug", "problem", "report", "reproduce", "reproduction",
    "steps", "step", "happened", "should", "workaround", "evidence",
    "station", "planet", "moon", "space", "location",
}

GENERIC_TOKENS = {
    "crash", "stuck", "cannot", "cant", "unable", "missing", "broken",
    "ui", "menu", "screen", "camera", "lag", "slow", "performance",
    "inventory", "equipment", "weapon", "weapons",
}

SCENARIO_SIGNALS = {
    "asop", "asopterminal", "store", "stored", "retrieve", "retrieved",
    "claim", "claimed", "impound", "impounded",
    "docking", "dockingport", "spindle", "grid", "cargo", "retract", "extended",
    "hangar", "pad", "elevator", "pershangar",
    "contract", "mission", "hauling", "haul", "distribution",
    "floor", "clipped", "clip", "spawn", "spawning",
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()

    # Hull C / Hull-C -> hullc
    t = re.sub(r"\bhull\s*-\s*([a-z0-9]+)\b", r"hull\1", t)
    t = re.sub(r"\bhull\s+([a-z0-9]+)\b", r"hull\1", t)

    t = re.sub(r"\basop\s*terminal\b", "asopterminal", t)
    t = re.sub(r"\bdocking\s*ports?\b", "dockingport", t)
    t = re.sub(r"\bpersonal\s*hangar\b", "pershangar", t)
    t = re.sub(r"\bpersistent\s*hangar\b", "pershangar", t)
    t = re.sub(r"\bpers\s*hangar\b", "pershangar", t)

    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()

def tokenize(text: str) -> List[str]:
    t = normalize_text(text)
    toks = TOKEN_RE.findall(t)
    return [x for x in toks if x and x not in STOPWORDS and len(x) >= 3]

def bigrams(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return []
    out = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a in STOPWORDS or b in STOPWORDS:
            continue
        out.append(f"{a}_{b}")
    return out
