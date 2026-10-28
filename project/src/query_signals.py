"""Pure text-signal helpers shared by serving and offline evaluation."""

from __future__ import annotations

from difflib import SequenceMatcher
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


BROAD_QUERY_TERMS = {
    "apparel",
    "clothes",
    "clothing",
    "fashion",
    "item",
    "items",
    "outfit",
    "outfits",
    "product",
    "products",
    "something",
    "style",
    "styles",
    "thing",
    "things",
    "wear",
}


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


def fuzzy_category_suggestions(text, categories, threshold=0.78, limit=3):
    """Suggest likely categories for misspelled query fragments.

    Suggestions never imply that the category should be applied automatically.
    The caller must ask the user to confirm a corrected query.
    """
    normalized_query = normalize_words(text)
    query_words = normalized_query.split()
    if not query_words:
        return []

    ranked = []
    for category in categories:
        best = None
        comparison_aliases = set(category_aliases(category))
        # A final noun such as "shoe" is useful for typo suggestions even
        # when it would be too ambiguous for exact hard filtering.
        comparison_aliases.update(
            alias.split()[-1]
            for alias in tuple(comparison_aliases)
            if len(alias.split()[-1]) >= 4
        )

        for alias in comparison_aliases:
            alias_words = alias.split()
            width = len(alias_words)
            if not alias or len(alias.replace(" ", "")) < 4:
                continue
            if width > len(query_words):
                windows = [(0, len(query_words), " ".join(query_words))]
            else:
                windows = [
                    (start, start + width, " ".join(query_words[start:start + width]))
                    for start in range(len(query_words) - width + 1)
                ]

            for start, end, fragment in windows:
                if fragment == alias:
                    continue
                # Generic words such as "clothes" and "fashion" describe the
                # search domain, not a misspelled category.  Comparing them to
                # catalog labels can otherwise produce false corrections such
                # as clothes -> Clutches.
                if all(
                    word in BROAD_QUERY_TERMS
                    for word in fragment.split()
                ):
                    continue
                score = SequenceMatcher(None, fragment, alias).ratio()
                candidate = (score, start, end)
                if best is None or candidate > best:
                    best = candidate

        if best is None or best[0] < float(threshold):
            continue

        score, start, end = best
        corrected_words = query_words[:start] + [str(category)] + query_words[end:]
        ranked.append({
            "category": str(category),
            "corrected_query": " ".join(corrected_words),
            "confidence": float(score),
        })

    ranked.sort(key=lambda item: (-item["confidence"], item["category"]))
    if not ranked:
        return []

    # Keep close alternatives for ambiguous nouns such as "shose", which can
    # reasonably refer to several shoe categories.
    best_score = ranked[0]["confidence"]
    return [
        item for item in ranked
        if item["confidence"] >= best_score - 0.03
    ][:int(limit)]


def is_broad_query(text):
    """Return whether a query asks for fashion items without a product type."""
    words = set(normalize_words(text).split())
    return bool(words & BROAD_QUERY_TERMS)


def categories_are_related(first, second):
    """Treat a broad category and its subtype as compatible."""
    left = normalize_words(first).replace(" ", "")
    right = normalize_words(second).replace(" ", "")
    return left == right or left.endswith(right) or right.endswith(left)
