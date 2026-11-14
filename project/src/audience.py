"""Conservative audience metadata for catalog browsing and explicit filters.

Audience labels are derived from catalog metadata only. The system never
infers a person's age or gender from an uploaded image. Product names are the
strongest evidence, followed by category. Descriptions are used only when they
contain an explicit audience phrase.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


AUDIENCE_ORDER = ("women", "men", "kids", "unisex", "unknown")
AUDIENCE_PREFERENCES = ("women", "men", "kids", "unisex")
AUDIENCE_QUERY_CONTEXT = {
    "women": "women's fashion product",
    "men": "men's fashion product",
    "kids": "children's fashion product",
    "unisex": "unisex fashion product",
}


@dataclass(frozen=True)
class AudienceInference:
    """Structured audience metadata and the evidence behind it."""

    audience: str
    age_group: str
    gender_group: str
    source: str
    confidence: float
    evidence: tuple[str, ...] = ()


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(
        character for character in value if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


# These are product styles, not statements that a product is for children.
_ADULT_STYLE_PHRASES = re.compile(
    r"\b(?:baby\s*dolls?|babydolls?|boy\s*legs?|boy\s*shorts?)\b"
)
_UNISEX = re.compile(r"\b(?:unisex|gender\s*neutral|genderless)\b")
_WOMEN = re.compile(r"\b(?:woman|women|womens|female|lady|ladies|for\s+her)\b")
_MEN = re.compile(
    r"\b(?:man|men|mens|male|gent|gents|gentleman|gentlemen|for\s+him)\b"
)
_BOYS = re.compile(r"\b(?:boy|boys|boyswear)\b")
_GIRLS = re.compile(r"\b(?:girl|girls|girlswear)\b")
_KIDS_STRONG = re.compile(
    r"\b(?:kid|kids|kidwear|kidswear|child|children|infant|infants|"
    r"toddler|toddlers|baby|babies|boy|boys|boyswear|girl|girls|girlswear)\b"
)
_KIDS_WEAK = re.compile(r"\b(?:junior|juniors|youth|teen|teens|teenage)\b")

# Long descriptions are noisy. Lone words such as "kids" or "junior" are not
# sufficient; only explicit audience constructions are accepted as fallback.
_DESCRIPTION_KIDS = re.compile(
    r"\b(?:designed|made|suitable|ideal|intended)\s+for\s+"
    r"(?:babies|infants|toddlers|children|boys|girls)\b|"
    r"\bfor\s+(?:babies|infants|toddlers|children|boys|girls)\s+"
    r"(?:aged|age|wear|clothing|fashion)\b"
)
_DESCRIPTION_WOMEN = re.compile(
    r"\b(?:designed|made|suitable|ideal|intended)\s+for\s+(?:women|ladies)\b"
)
_DESCRIPTION_MEN = re.compile(
    r"\b(?:designed|made|suitable|ideal|intended)\s+for\s+(?:men|gentlemen)\b"
)
_DESCRIPTION_UNISEX = re.compile(
    r"\b(?:designed|made|suitable|ideal|intended)\s+for\s+"
    r"(?:all\s+genders|men\s+and\s+women|women\s+and\s+men)\b"
)


def _signals(text: str) -> dict[str, bool]:
    cleaned = _ADULT_STYLE_PHRASES.sub(" ", _normalise(text))
    return {
        "unisex": bool(_UNISEX.search(cleaned)),
        "women": bool(_WOMEN.search(cleaned)),
        "men": bool(_MEN.search(cleaned)),
        "kids": bool(_KIDS_STRONG.search(cleaned) or _KIDS_WEAK.search(cleaned)),
        "boys": bool(_BOYS.search(cleaned)),
        "girls": bool(_GIRLS.search(cleaned)),
    }


def _structured_result(text: str, source: str, confidence: float):
    signals = _signals(text)
    evidence = tuple(name for name, present in signals.items() if present)
    if not evidence:
        return None

    # "Unisex kids" is valid. Adult gender wording mixed with child wording is
    # ambiguous and must not be forced into the Kids page.
    if signals["kids"] and (signals["women"] or signals["men"]):
        return AudienceInference(
            "unknown", "unknown", "unknown", f"conflicting_{source}", 0.25,
            evidence,
        )

    if signals["kids"]:
        if signals["boys"] and signals["girls"]:
            gender = "unisex"
        elif signals["boys"]:
            gender = "men"
        elif signals["girls"]:
            gender = "women"
        elif signals["unisex"]:
            gender = "unisex"
        else:
            gender = "unknown"
        return AudienceInference("kids", "kids", gender, source, confidence, evidence)

    if signals["unisex"] or (signals["women"] and signals["men"]):
        return AudienceInference(
            "unisex", "adult", "unisex", source, confidence, evidence
        )
    if signals["women"]:
        return AudienceInference(
            "women", "adult", "women", source, confidence, evidence
        )
    if signals["men"]:
        return AudienceInference("men", "adult", "men", source, confidence, evidence)
    return None


def _description_result(description: str):
    text = _ADULT_STYLE_PHRASES.sub(" ", _normalise(description))
    evidence = []
    if _DESCRIPTION_KIDS.search(text):
        evidence.append("kids")
    if _DESCRIPTION_WOMEN.search(text):
        evidence.append("women")
    if _DESCRIPTION_MEN.search(text):
        evidence.append("men")
    if _DESCRIPTION_UNISEX.search(text):
        evidence.append("unisex")
    if not evidence:
        return None
    if "kids" in evidence and any(item in evidence for item in ("women", "men")):
        return AudienceInference(
            "unknown", "unknown", "unknown", "conflicting_description", 0.2,
            tuple(evidence),
        )
    if "kids" in evidence:
        return AudienceInference(
            "kids", "kids", "unknown", "description", 0.65, tuple(evidence)
        )
    if "unisex" in evidence or ("women" in evidence and "men" in evidence):
        return AudienceInference(
            "unisex", "adult", "unisex", "description", 0.6, tuple(evidence)
        )
    gender = "women" if "women" in evidence else "men"
    return AudienceInference(
        gender, "adult", gender, "description", 0.6, tuple(evidence)
    )


def infer_audience(
    product_name: str = "",
    description: str = "",
    category: str = "",
) -> AudienceInference:
    """Infer age and gender using product name, category, then description."""
    for value, source, confidence in (
        (product_name, "product_name", 1.0),
        (category, "category", 0.9),
    ):
        result = _structured_result(value, source, confidence)
        if result is not None:
            return result

    result = _description_result(description)
    if result is not None:
        return result
    return AudienceInference(
        "unknown", "unknown", "unknown", "insufficient_metadata", 0.0, ()
    )


def normalise_audience_preference(value: str | None) -> str | None:
    """Validate an optional audience explicitly selected by the user."""
    if value is None or not str(value).strip():
        return None
    audience = str(value).strip().casefold()
    if audience not in AUDIENCE_PREFERENCES:
        raise ValueError("audience must be one of: women, men, kids, unisex")
    return audience


def add_audience_context(text: str, audience: str | None) -> str:
    """Add a short semantic qualifier without changing the visible query."""
    audience = normalise_audience_preference(audience)
    if audience is None:
        return str(text).strip()
    return f"{str(text).strip()}, {AUDIENCE_QUERY_CONTEXT[audience]}".strip(", ")
