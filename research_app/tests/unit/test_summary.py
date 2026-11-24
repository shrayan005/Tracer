import json

import pytest

from app.domain.errors import ProviderRejected
from app.domain.schemas import PaperRecord, PaperSummary
from app.integrations.llm.gateway import LlmGateway
from app.services.fulltext_context import PaperText
from app.services.summary_service import SummaryService, check, quote_found, ungrounded_numbers
from tests.fakes import FakeProvider

ABSTRACT = ("We measured triceps brachii activation in 24 trained men during bench press at 70% of "
            "one-repetition maximum. The long head showed 18.5% higher activation with a narrow grip. "
            "The study was limited to men.")
QUOTE = "The long head showed 18.5% higher activation with a narrow grip."


def summary_json(findings=None) -> str:
    findings = findings if findings is not None else [
        {"statement": "Long head activation was 18.5% higher with a narrow grip", "quote": QUOTE}]
    return json.dumps({"tldr": "Grip changes triceps activation.", "objective": "Compare grips.",
                       "methods": "24 trained men, EMG.", "key_findings": findings,
                       "limitations": ["only men"], "why_relevant": "Directly about triceps."})


def make_service(script: list) -> tuple[SummaryService, FakeProvider]:
    provider = FakeProvider("a", script)

    async def sink(_record):
        pass

    async def no_sleep(_s):
        pass

    gateway = LlmGateway([provider], usage_sink=sink, max_retries=0, sleep=no_sleep)
    return SummaryService(gateway, max_parallel=2), provider


def paper(title="Triceps and grip width", abstract=ABSTRACT) -> PaperRecord:
    return PaperRecord(source="x", title=title, abstract=abstract, year=2022)


# ---------- abstract only ----------

async def test_valid_reply_becomes_a_summary():
    service, provider = make_service([summary_json()])
    results, warnings = await service.summarize("triceps", {7: paper()})
    assert warnings == []
    assert results[7].based_on == "abstract"
    assert results[7].summary.key_findings[0].quote == QUOTE
    assert results[7].flags == []
    assert provider.calls == 1
    prompt = provider.last_kwargs["messages"][0]["content"]
    assert "title and abstract" in prompt and "COPIED EXACTLY" in prompt


async def test_paper_with_no_abstract_and_no_full_text_skips_the_llm():
    service, provider = make_service([])
    results, _ = await service.summarize("triceps", {0: paper(abstract=None)})
    assert results[0].based_on == "title only" and results[0].summary is None
    assert provider.calls == 0


async def test_one_failure_does_not_stop_the_others():
    service, _ = make_service([ProviderRejected("a", "bad request"), summary_json()])
    results, warnings = await service.summarize("triceps", {0: paper("First paper"), 1: paper("Second paper")})
    assert len(results) == 1
    assert len(warnings) == 1 and "could not summarise" in warnings[0]


# ---------- full text ----------

FULL = PaperText(context="## Methods\nTen women trained for 8 weeks.\n\n## Results\nStrength rose by 23.4%.",
                 source="Ten women trained for 8 weeks. Strength rose by 23.4% after training.")


async def test_full_text_is_used_when_available():
    finding = {"statement": "Strength rose 23.4%", "quote": "Strength rose by 23.4% after training"}
    service, provider = make_service([summary_json([finding])])
    results, _ = await service.summarize("triceps", {3: paper(abstract=None)}, texts={3: FULL})
    assert results[3].based_on == "full text" and results[3].flags == []
    messages = provider.last_kwargs["messages"]
    assert "sections of the full text" in messages[0]["content"]
    assert "## Methods\nTen women trained" in messages[1]["content"]       # the context went into the prompt


async def test_invented_quote_and_number_are_flagged():
    finding = {"statement": "Strength rose 42%", "quote": "Strength doubled in every participant"}
    service, _ = make_service([summary_json([finding])])
    results, _ = await service.summarize("triceps", {3: paper()}, texts={3: FULL})
    flags = results[3].flags
    assert any(f.startswith("quote not found in the paper") for f in flags)
    assert "number not found in the paper: 42" in flags


# ---------- the code checks ----------

def test_quote_found_tolerates_formatting_but_not_paraphrase():
    source = "The long head showed **18.5%** higher activation — with a narrow grip (n = 24)."
    assert quote_found("the long head showed 18.5% higher activation - with a narrow grip", source)
    assert quote_found("“The long head showed 18.5% higher activation”", source)          # curly quotes
    assert quote_found("The long head shoed 18.5% higher activation — with a narrow grip (n = 24)", source)  # 1 typo
    assert not quote_found("The long head was much more active when the grip was narrow", source)   # paraphrase
    assert not quote_found("The long head showed 18.5% higher activation in all of the women tested", source)  # half invented
    assert quote_found("the ﬁnal grip", "The final grip was used.")                                   # PDF ligature
    assert not quote_found("", source)


def test_missing_quote_is_flagged():
    summary = PaperSummary.model_validate_json(summary_json([{"statement": "Something", "quote": None}]))
    assert check(summary, ABSTRACT) == ["no quote given for: Something"]


def test_numbers_check_accepts_real_and_small_numbers():
    summary = PaperSummary.model_validate_json(summary_json([
        {"statement": "24 men at 70% 1RM", "quote": QUOTE}, {"statement": "18.5% higher in 2 grips", "quote": QUOTE}]))
    assert ungrounded_numbers(summary, ABSTRACT) == []


def test_old_saved_summaries_with_plain_string_findings_still_load():
    old = {"tldr": "t", "objective": "o", "methods": "m", "key_findings": ["Old finding one", "Old two"],
           "limitations": [], "why_relevant": "w"}
    summary = PaperSummary.model_validate(old)
    assert [f.statement for f in summary.key_findings] == ["Old finding one", "Old two"]
    assert all(f.quote is None for f in summary.key_findings)


@pytest.mark.parametrize("bad", [[], [{"statement": f"f{i}", "quote": "q"} for i in range(6)]])
def test_finding_count_is_limited(bad):
    with pytest.raises(ValueError):
        PaperSummary.model_validate_json(summary_json(bad))


def test_stitched_quote_is_flagged():
    finding = {"statement": "Strength rose", "quote": "Strength rose by 23.4% [...] after training"}
    summary = PaperSummary.model_validate_json(summary_json([finding]))
    assert check(summary, "Strength rose by 23.4% after training") == [
        'quote joins separate passages: "Strength rose by 23.4% [...] after training"']


def test_quotes_match_text_with_markdown_italics():
    # measured (6.7): extracted text keeps italics as "_p_" and "_η_2"; the model quotes clean text
    source = "ANCOVA revealed no difference for these changes between groups ( _p_ = 0.640, _η_2 = 0.015)."
    assert quote_found("ANCOVA revealed no difference for these changes between groups (p = 0.640, η2 = 0.015).",
                       source)
    assert quote_found("The authors declare no conflict of interest", "**Conflicts of Interest:** The authors "
                       "declare no conflict of interest.")
    assert not quote_found("ANCOVA revealed a large difference between the groups (p = 0.001, η2 = 0.4).", source)
