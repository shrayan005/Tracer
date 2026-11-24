"""
Grounding checks (shared by summaries and paper analysis): is what the LLM wrote really in the paper?

  quote_found()       a quote must appear in the paper: exactly (after normalising quotes, dashes,
                      Markdown and spaces) or near-exactly (word by word, tolerating a garbled word)
  numbers_missing()   numbers the LLM wrote must appear somewhere in the paper text
"""

import re
from difflib import SequenceMatcher

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
SMALL_NUMBER = 10               # "two groups" may come back as "2 groups": don't flag small integers
_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                        "−": "-", " ": " ", "ﬁ": "fi", "ﬂ": "fl"})   # PDF ligatures ﬁ ﬂ


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in NUMBER.findall(text)}


def numbers_missing(texts: list[str], source: str) -> list[str]:
    """Numbers in `texts` that do NOT appear anywhere in the source (small integers are allowed)."""
    known = _numbers(source)
    missing: list[str] = []
    for text in texts:
        for raw in NUMBER.findall(text):
            value = float(raw.replace(",", "."))
            if value in known or (value.is_integer() and value <= SMALL_NUMBER):
                continue
            missing.append(raw)
    return missing


def normalize(text: str) -> str:
    """For comparing quotes: lower case, plain quotes/dashes, no Markdown marks, single spaces."""
    text = text.translate(_PUNCT).lower()
    text = re.sub(r"[*_`]", "", text)          # emphasis marks are deleted: "_η_2" -> "η2" (measured 6.7)
    text = re.sub(r"[#|]", " ", text)          # headings / table bars separate words
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([(\[]) ", r"\1", re.sub(r" ([)\]])", r"\1", text))   # "( p = 1 )" -> "(p = 1)"
    return text.strip(" .\"'")


QUOTE_MATCH = 0.85             # word-by-word similarity needed for a near-exact quote


def quote_found(quote: str, source: str) -> bool:
    """
    Exact (after normalising), or near-exact: find where the quote starts in the paper (a shared
    3-word run), then compare word by word. One garbled word in a 14-word quote still scores ~0.9;
    a paraphrase or a half-invented quote scores far lower.
    """
    q, s = normalize(quote), normalize(source)
    if not q:
        return False
    if q in s:
        return True
    words, source_words = q.split(), s.split()
    if len(words) < 6:                                    # too short to judge fuzzily: exact only
        return False
    starts: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(source_words) - 2):
        starts.setdefault(tuple(source_words[i:i + 3]), []).append(i)
    best = 0.0
    for j in range(len(words) - 2):
        for i in starts.get(tuple(words[j:j + 3]), [])[:20]:
            window = source_words[max(0, i - j): max(0, i - j) + len(words) + 2]
            best = max(best, SequenceMatcher(None, words, window).ratio())
    return best >= QUOTE_MATCH
