"""The reliability harness: source grounding, confidence caps, majority voting, page reading."""
from types import SimpleNamespace

import pytest

from farm.orchestrator import Farm, cap_confidence, ground, norm_url
from farm.pages import html_to_text, relevant_excerpt


def finding(url, confidence="high"):
    return {"source_url": url, "confidence": confidence}


def test_ground_grades_findings_by_what_the_agent_saw():
    sources = [{"url": "https://www.a.com/page/", "opened": True},
               {"url": "https://b.com/x", "cited": True},
               {"url": "https://c.com/y"}]
    findings = [finding("http://a.com/page"), finding("https://b.com/x"), finding("https://c.com/y"),
                finding("https://made-up.example/z")]
    ground(findings, sources)
    assert [f["grounding"] for f in findings] == ["read", "read", "seen", "unverified"]


@pytest.mark.parametrize("grounding, model, capped", [
    ("verified", "high", "high"), ("read", "high", "high"), ("seen", "high", "medium"),
    ("seen", "low", "low"), ("unverified", "high", "low"), ("disputed", "medium", "low"),
])
def test_confidence_is_capped_by_source_quality(grounding, model, capped):
    f = {"confidence": model, "grounding": grounding}
    cap_confidence(f)
    assert f["confidence"] == capped
    assert f["model_confidence"] == model


def test_norm_url_ignores_scheme_www_case_and_trailing_slash():
    assert norm_url("HTTPS://www.Example.com/Path/") == norm_url("http://example.com/path")


def gap(title, impact="high"):
    return {"title": title, "description": title, "why_unsolved": "", "opportunity": "",
            "attack_ideas": [f"idea: {title}"], "evidence": ["F1"], "impact": impact,
            "difficulty": "medium", "status": "new"}


class FakeMerger:
    mock = False

    def __init__(self, groups):
        self.groups = groups

    def ask(self, role, system, prompt, schema=None, web=None):
        return SimpleNamespace(data={"groups": self.groups})


def test_majority_vote_keeps_gaps_most_passes_agree_on():
    passes = [[gap("field accuracy"), gap("offline size")],
              [gap("field accuracy, reworded"), gap("trust")],
              [gap("field accuracy again"), gap("small offline models"), gap("pricing", "low")]]
    merger = FakeMerger([{"members": ["P1.1", "P2.1", "P3.1"], "representative": "P2.1"},
                         {"members": ["P1.2", "P3.2"], "representative": "P1.2"}])
    kept = Farm(None, merger, questions=3, workers=1, log=lambda m: None)._vote(passes)

    assert [(g["title"], g["votes"]) for g in kept] == [("field accuracy, reworded", 3), ("offline size", 2)]
    assert kept[0]["attack_ideas"] == ["idea: field accuracy", "idea: field accuracy, reworded",
                                       "idea: field accuracy again"]


def test_majority_vote_falls_back_to_best_supported_when_nothing_agrees():
    passes = [[gap("a")], [gap("b")], [gap("c")]]
    kept = Farm(None, FakeMerger([]), questions=3, workers=1, log=lambda m: None)._vote(passes)
    assert len(kept) == 3 and all(g["votes"] == 1 for g in kept)


def test_html_to_text_drops_scripts_and_keeps_blocks():
    html = ("<html><head><title>t</title><style>p{}</style></head><body><nav>Menu</nav>"
            "<script>var x = 1;</script><h1>MobileNetV2</h1><p>Inference  under\t200 ms.</p></body></html>")
    assert html_to_text(html) == "Menu\nMobileNetV2\nInference under 200 ms."


def test_relevant_excerpt_keeps_the_chunks_that_match_the_claims():
    filler = "Unrelated text about something else entirely. " * 60
    page = filler + "The model uses MobileNetV2 with TFLite on Android phones. " + filler
    excerpt = relevant_excerpt(page, ["Uses MobileNetV2 TFLite backbone"], budget=1300, chunk=1200)
    assert "MobileNetV2" in excerpt
    assert len(excerpt) < len(page)
