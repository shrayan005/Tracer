from app.services.fulltext_context import BUDGET, build_context, paper_text, trim


def test_trim_cuts_at_a_sentence_end():
    text = "First sentence is here. Second sentence is here. Third sentence is quite a bit longer."
    assert trim(text, 1000) == text
    assert trim(text, 55) == "First sentence is here. Second sentence is here. [...]"


def test_trim_without_sentence_end_cuts_at_a_space():
    assert trim("one two three four five six", 12) == "one two [...]"


def test_only_useful_kinds_are_kept_in_budget_order():
    sections = [("front", "Title", "Authors and affiliations"), ("introduction", "Intro", "Background text."),
                ("results", "Results", "Force rose."), ("methods", "Methods", "Ten women."),
                ("references", "References", "1. Smith"), ("caption", "Fig 1", "A figure")]
    context = build_context(sections)
    assert context == "## Methods\nTen women.\n\n## Results\nForce rose."     # methods before results
    assert "Authors" not in context and "Smith" not in context


def test_each_kind_is_trimmed_to_its_budget():
    long_methods = "Participants did the task. " * 1000                     # ~27k characters
    context = build_context([("methods", "Methods", long_methods)])
    assert context.endswith("[...]")
    assert len(context) <= len("## Methods\n") + BUDGET["methods"] + len(" [...]")


def test_limitation_paragraphs_are_moved_out_of_the_discussion_before_trimming():
    discussion = ("Our findings agree with earlier work. " * 120 +        # long: would be trimmed away ...
                  "\n\nA limitation of this study is the small sample of ten women.")   # ... with this at the end
    context = build_context([("discussion", "Discussion", discussion)])
    assert "## Limitations\nA limitation of this study is the small sample of ten women." in context
    assert context.index("## Discussion") < context.index("## Limitations")


def test_tables_keep_their_title():
    context = build_context([("table", "Table 1 Force by angle", "| Angle | Force |")])
    assert context == "## Table\n### Table 1 Force by angle\n| Angle | Force |"


def test_paper_text_keeps_everything_as_the_source_for_checks():
    text = paper_text([("methods", "Methods", "Ten women."), ("references", "References", "1. Smith")])
    assert text.context == "## Methods\nTen women."
    assert text.source == "Ten women.\n\n1. Smith"                          # quotes are checked against ALL text


def test_statements_are_rescued_from_any_section_for_the_analysis():
    # measured (6.4): the conflict statement sat at the END of "Conclusions", the data statement after the
    # references in the backmatter; trimming to the budget cut both
    sections = [("conclusion", "Conclusions", "Training worked. " * 100 + "No grant support was received for "
                 "this study. Conflicts of Interest: The authors declare no conflict of interest."),
                ("references", "References", "1. Smith J. Conflicts of interest in sport. 2019."),
                ("backmatter", "Notes", "x " * 900 + "The datasets for this study will not be made publicly "
                 "available because of ethics rules.")]
    context = build_context(sections, {"conclusion": 800, "backmatter": 300, "statements": 1000})
    part = context[context.index("## Statements"):]
    assert "No grant support was received for this study." in part
    assert "Conflicts of Interest: The authors declare no conflict of interest." in part
    assert "will not be made publicly available" in part
    assert "Smith J." not in part                                     # references are never searched
    assert "## Statements" not in build_context(sections)             # the summary budget has no statements


def test_the_statistics_subsection_gets_its_own_budget_for_the_analysis():
    # measured (6.7): "Statistical Analysis" is the last methods subsection, so a long Methods trim cut it
    sections = [("methods", "Participants", "Nine women took part. " * 400),
                ("methods", "Statistical Analysis", "Partial eta squared was used as an effect size.")]
    analysis = build_context(sections, {"methods": 500, "statistics": 300})
    assert "## Statistics\nPartial eta squared was used as an effect size." in analysis
    assert "eta squared" not in build_context(sections, {"methods": 500})   # other budgets: unchanged
