"""Provider plumbing that doesn't need a network: citations, rate limits, model fallback."""
from types import SimpleNamespace

import pytest

from farm import config
from farm.llm import LLM, Meter, QuotaExhausted, _groq_sources, _retry_after


def test_groq_citations_become_urls_of_opened_pages():
    tools = [{"index": 0, "type": "browser_search",
              "search_results": {"results": [{"url": "https://a.com", "title": "A"}]}},
             {"index": "2", "type": "browser.open",
              "search_results": {"results": [{"url": "https://b.com/page", "title": "x"}]}}]
    text, sources, searches = _groq_sources("Claim 【2†L1-L4】 and 【0†L9】.", tools)
    assert text == "Claim  (https://b.com/page) and ."
    assert searches == 1
    assert {s["url"]: (s.get("opened", False), s.get("cited", False)) for s in sources} == {
        "https://a.com": (False, False), "https://b.com/page": (True, True)}


@pytest.mark.parametrize("headers, seconds", [
    ({"retry-after": "7"}, 8.0), ({"x-ratelimit-reset-tokens": "1m26.4s"}, 87.4), ({}, 30.0)])
def test_retry_after_reads_groq_headers(headers, seconds):
    assert _retry_after(headers) == pytest.approx(seconds)


def test_groq_moves_to_the_next_model_when_a_daily_quota_runs_out(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = LLM(Meter(None), log=lambda m: None)
    tried = []

    def fake_call(role, model, system, prompt, schema, web):
        tried.append(model)
        if model != "openai/gpt-oss-120b":
            raise QuotaExhausted(model)
        message = SimpleNamespace(content='{"ok": 1}', model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    llm._groq_call = fake_call
    assert llm.ask("researcher", "s", "p", schema={"type": "object"}, web={}).data == {"ok": 1}
    assert tried == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

    tried.clear()
    llm.ask("researcher", "s", "p")
    assert tried == ["openai/gpt-oss-120b"]  # the exhausted model is skipped from now on

    llm._groq_call = lambda *a: (_ for _ in ()).throw(QuotaExhausted("all"))
    with pytest.raises(QuotaExhausted):
        llm.ask("planner", "s", "p")


def test_price_lookup_prefers_the_longest_matching_model_name():
    assert config.price_for("claude-opus-5-5") == (4.0, 20.0)
    assert config.price_for("claude-opus-5") == (5.0, 25.0)
