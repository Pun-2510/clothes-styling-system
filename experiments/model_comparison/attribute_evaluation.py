"""Controlled attribute vocabulary used only by legacy research experiments.

The deployed project ranking does not import this module. Runtime text-bearing
queries use the catalog-derived TF-IDF index instead of a fixed word list.
"""

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
    ("patterned", (r"\b(?:patterned|printed|graphic|motif)\b",),
     (r"\b(?:patterned|printed|graphic|pattern|striped|check|checked|plaid|floral|polka)\b",)),
    ("striped", (r"\b(?:stripe|striped|stripes)\b",),
     (r"\b(?:stripe|striped|stripes)\b",)),
    ("checked", (r"\b(?:check|checked|checkered|plaid)\b",),
     (r"\b(?:check|checked|checkered|plaid)\b",)),
    ("floral", (r"\b(?:floral|flower|flowers)\b",),
     (r"\b(?:floral|flower|flowers)\b",)),
)


def extract_query_attributes(text):
    query = str(text).lower()
    return [
        name
        for name, query_patterns, _ in ATTRIBUTE_RULES
        if any(re.search(pattern, query) for pattern in query_patterns)
    ]
