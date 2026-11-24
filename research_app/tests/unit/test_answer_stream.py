"""SSE step 4: reading the answer JSON while it is written, and checking each sentence as soon as it is finished.
Fakes only: no LLM, no database."""

import json

from app.integrations.llm.gateway import LlmGateway
from app.services.answer_stream import DraftReader, Finished, Growing
from app.services.ask_service import AskService
from tests.fakes import FakeProvider
from tests.unit.test_ask import PAPERS, hit

QUOTE = "resulted in significantly greater increases in fat-free mass"
ANSWER = json.dumps({"sentences": [
    {"text": "Brose et al. found greater fat-free mass gains.", "excerpt": "E1", "quote": QUOTE},
    {"text": "Lean mass rose more with creatine.", "excerpt": "E1", "quote": QUOTE.upper()},      # a repeat
    {"text": "Gains were 40% larger.", "excerpt": "E1", "quote": "forty percent larger"},       # fails its check
], "answered": True, "missing": None})


def feed_in_pieces(text: str, size: int) -> list:
    reader = DraftReader()
    news = []
    for i in range(0, len(text), size):
        news += reader.feed(text[i:i + size])
    return news


def test_text_grows_and_each_sentence_finishes_once_in_order():
    news = feed_in_pieces(ANSWER, size=7)
    finished = [n for n in news if isinstance(n, Finished)]
    assert [n.index for n in finished] == [0, 1, 2]
    assert finished[0].sentence.quote == QUOTE
    growing = [n.text for n in news if isinstance(n, Growing) and n.index == 0]
    assert len(growing) > 3 and all(b.startswith(a) for a, b in zip(growing, growing[1:]))   # only grows
    # a sentence is never Finished before its quote is complete
    first_done = news.index(finished[0])
    assert not any(isinstance(n, Growing) and n.index == 0 for n in news[first_done:])


def test_a_finished_sentence_needs_its_quote_closed_whatever_the_field_order():
    reader = DraftReader()
    assert not [n for n in reader.feed('{"sentences":[{"quote":"a b","excerpt":"E1","text":"Some') if isinstance(n, Finished)]
    news = reader.feed(' text."}')
    assert [n.sentence.text for n in news if isinstance(n, Finished)] == ["Some text."]


def test_a_sentence_missing_a_field_still_finishes_when_the_model_moves_on():
    reader = DraftReader()
    reader.feed('```json\n{"sentences":[{"text":"No quote here.","excerpt":"E1"}')
    news = reader.feed(',{"text":"Next')                    # the model started the next sentence
    done = [n for n in news if isinstance(n, Finished)]
    assert done and done[0].sentence.quote == "" and done[0].sentence.text == "No quote here."


def test_answered_written_before_the_list_does_not_finish_sentences_early():
    # measured live (Gemini): {"answered": true, "missing": null, "sentences": [...]} in that order
    first = json.loads(ANSWER)
    reordered = json.dumps({"answered": True, "missing": None, "sentences": first["sentences"]}, indent=2)
    finished = [n for n in feed_in_pieces(reordered, size=40) if isinstance(n, Finished)]
    assert [n.sentence.quote for n in finished] == [s["quote"] for s in first["sentences"]]


def stream_service(*script) -> AskService:
    return AskService(unit_of_work=None, retrieval=None, gateway=LlmGateway([FakeProvider("a", list(script))]))


async def run(service: AskService) -> list[tuple[str, object]]:
    return [e async for e in service._stream_draft("Did creatine help?", [hit()], PAPERS, [])]


async def test_each_sentence_is_streamed_then_sent_checked():
    events = await run(stream_service(ANSWER))
    kinds = [e for e, _ in events]
    assert kinds[0] == "text" and kinds[-1] == "draft" and "reset" not in kinds
    sent = [d for e, d in events if e == "sentence"]
    assert sent[0]["part"]["citation"]["rank"] == 4 and sent[0]["part"]["flags"] == []
    assert sent[1]["part"] is None                                   # the repeated point is dropped
    assert "quote not found in the passage" in sent[2]["part"]["flags"]
    assert len(events[-1][1].sentences) == 3                          # the whole draft, for saving


async def test_invalid_streamed_json_resets_and_asks_again_without_streaming():
    events = await run(stream_service("I think creatine helps.", ANSWER))
    kinds = [e for e, _ in events]
    assert kinds == ["reset", "draft"]                               # nothing parseable was shown
    assert events[-1][1].sentences[0].excerpt == "E1"
