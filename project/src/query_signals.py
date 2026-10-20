"""Pure text-signal helpers shared by serving and offline evaluation."""

from __future__ import annotations

import re


ATTRIBUTE_RULES = (
    ("black", (r"\bblack\b",), (r"\bblack\b",)),
    ("white", (r"\bwhite\b",), (r"\bwhite\b",)),
    ("red", (r"\bred\b",), (r"\bred\b",)),
    ("blue", (r"\bblue\b",), (r"\bblue\b",)),
    ("green", (r"\bgreen\b",), (r"\bgreen\b",)),
    ("yellow", (r"\byellow\b",), (r"\byellow\b",)),
    ("pink", (r"\bpink\b",), (r"\bpink\b",)),
    ("orange", (r"\borange\b",), (r"\borange\b",)),
    ("purple", (r"\bpurple\b",), (r"\bpurple\b",)),
    ("brown", (r"\bbrown\b",), (r"\bbrown\b",)),
    ("grey", (r"\b(?:grey|gray)\b",), (r"\b(?:grey|gray)\b",)),
    ("navy", (r"\bnavy\b",), (r"\bnavy\b",)),
    ("maroon", (r"\bmaroon\b",), (r"\bmaroon\b",)),
    ("beige", (r"\bbeige\b",), (r"\bbeige\b",)),
    ("gold", (r"\b(?:gold|golden)\b",), (r"\b(?:gold|golden)\b",)),
    ("silver", (r"\bsilver\b",), (r"\bsilver\b",)),
    (
        "patterned",
        (r"\b(?:patterned|printed|graphic|motif)\b",),
        (
            r"\b(?:patterned|printed|graphic|pattern|striped|check|checked|"
            r"plaid|floral|polka)\b",
        ),
    ),
    (
        "striped",
        (r"\b(?:stripe|striped|stripes)\b",),
        (r"\b(?:stripe|striped|stripes)\b",),
    ),
    (
        "checked",
        (r"\b(?:check|checked|checkered|plaid)\b",),
        (r"\b(?:check|checked|checkered|plaid)\b",),
    ),
    (
        "floral",
        (r"\b(?:floral|flower|flowers)\b",),
        (r"\b(?:floral|flower|flowers)\b",),
    ),
)


def extract_query_attributes(text):
    """Return explicit color or pattern attributes found in a query."""
    query = str(text).lower()
    return [
        name
        for name, query_patterns, _ in ATTRIBUTE_RULES
        if any(re.search(pattern, query) for pattern in query_patterns)
    ]


def normalize_words(value):
    """Normalize text for exact word-boundary category matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def category_aliases(category):
    """Return conservative singular and T-shirt aliases for one category."""
    normalized = normalize_words(category)
    aliases = {normalized}
    words = normalized.split()

    if words:
        last = words[-1]
        if last.endswith("ies") and len(last) > 3:
            singular = last[:-3] + "y"
        elif last.endswith("ses") and len(last) > 3:
            singular = last[:-2]
        elif last.endswith("s") and len(last) > 2:
            singular = last[:-1]
        else:
            singular = last
        aliases.add(" ".join(words[:-1] + [singular]))

    if "tshirts" in normalized:
        aliases.add(normalized.replace("tshirts", "t shirts"))
        aliases.add(normalized.replace("tshirts", "t shirt"))
    if normalized == "tshirts":
        aliases.update({"tee", "tees"})

    return {alias for alias in aliases if alias}


def infer_category_from_text(text, categories):
    """Find the longest explicitly named catalog category in a query."""
    normalized_query = normalize_words(text)
    matches = []
    for category in categories:
        for alias in category_aliases(category):
            if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
                matches.append((len(alias), str(category)))
    return max(matches)[1] if matches else None


def categories_are_related(first, second):
    """Treat a broad category and its subtype as compatible."""
    left = normalize_words(first).replace(" ", "")
    right = normalize_words(second).replace(" ", "")
    return left == right or left.endswith(right) or right.endswith(left)
