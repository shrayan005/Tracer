"""
Reading the LLM's answer while it is still being written (SSE step 4).

The model answers in JSON, and the JSON arrives in pieces:
    {"sentences":[{"text":"Brose et al. gave 5 g        <- unfinished
jiter (already installed: the openai package uses it) parses such unfinished JSON in two ways:
    trailing-strings   keeps a string that is still being written   -> the growing text, shown as it comes
    on                 drops it                                     -> only what is finished
A sentence is finished when its text, excerpt and quote are all closed strings (in any order), or when the
model has moved on to the next sentence or past the list. Only then can code check its quote.
"""

from dataclasses import dataclass
from typing import Any

import jiter

from app.domain.schemas import AnswerSentence

FIELDS = ("text", "excerpt", "quote")


@dataclass
class Growing:
    """Sentence `index` so far (its text is not finished: it may still grow)."""
    index: int
    text: str


@dataclass
class Finished:
    """Sentence `index` is complete: ready to check."""
    index: int
    sentence: AnswerSentence


class DraftReader:
    def __init__(self) -> None:
        self.buffer = ""
        self.finished = 0                     # sentences already reported as Finished
        self.shown: dict[int, str] = {}       # the text last reported per sentence (only report changes)

    def feed(self, piece: str) -> list[Growing | Finished]:
        """Add a piece of the model's output; return what is new since the last piece."""
        self.buffer += piece
        start = self.buffer.find("{")        # skip a ```json fence or a sentence before the JSON
        if start == -1:
            return []
        raw = self.buffer[start:].encode()
        try:
            growing = jiter.from_json(raw, partial_mode="trailing-strings")
            closed = jiter.from_json(raw, partial_mode="on")
        except ValueError:                     # not parseable yet (or not JSON at all): wait for more
            return []
        if not isinstance(growing, dict) or not isinstance(closed, dict):
            return []

        news: list[Growing | Finished] = []
        done = sentences(closed)
        # a key AFTER "sentences" means the list is done. Measured (live, Gemini): the model may also write
        # "answered"/"missing" BEFORE the list, so "any other key" is not enough: the order counts
        keys = list(closed)
        past_the_list = "sentences" in keys and keys.index("sentences") < len(keys) - 1
        while self.finished < len(done):
            s = done[self.finished]
            moved_on = self.finished < len(done) - 1 or past_the_list
            if not (all(isinstance(s.get(f), str) for f in FIELDS) or moved_on):
                break
            news.append(Finished(self.finished, to_sentence(s)))
            self.finished += 1

        for i, s in enumerate(sentences(growing)):
            if i < self.finished:
                continue
            text = s.get("text")
            if isinstance(text, str) and text and text != self.shown.get(i):
                self.shown[i] = text
                news.append(Growing(i, text))
        return news


def sentences(parsed: dict) -> list[dict]:
    items = parsed.get("sentences")
    return [s if isinstance(s, dict) else {} for s in items] if isinstance(items, list) else []


def to_sentence(s: dict[str, Any]) -> AnswerSentence:
    """A finished sentence; a field the model left out becomes empty (the check then flags it)."""
    return AnswerSentence(**{f: str(s.get(f) or "") for f in FIELDS})
