"""Thin wrapper around the AI provider (Claude, Grok or Groq) shared by every agent in the farm."""
import itertools
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

import anthropic

from . import config


class RefusalError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class QuotaExhausted(BudgetExceeded):
    """The provider's own usage limit ran out; the run pauses and can be resumed later."""


@dataclass
class Reply:
    text: str = ""
    data: dict | None = None
    sources: list[dict] = field(default_factory=list)


class Meter:
    """Thread-safe spend tracker with an optional hard ceiling."""

    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd or None
        self.cost = 0.0
        self.calls = 0
        self.searches = 0
        self._lock = threading.Lock()

    def record(self, model: str, tokens_in: float, tokens_out: int, searches: int) -> None:
        price_in, price_out = config.price_for(model)
        cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        cost += searches * config.WEB_SEARCH_PRICE[config.PROVIDER]
        with self._lock:
            self.cost += cost
            self.calls += 1
            self.searches += searches

    def check(self) -> None:
        if self.limit is not None and self.cost >= self.limit:
            raise BudgetExceeded(f"spent ${self.cost:.2f} of the ${self.limit:.2f} session limit")


def web_tools(searches: int, fetches: int) -> list[dict]:
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": searches}]
    if fetches:
        tools.append({"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": fetches})
    return tools


class LLM:
    def __init__(self, meter: Meter, mock: bool = False, log=print):
        self.meter = meter
        self.mock = mock
        self.log = log
        self.client = None
        self._exhausted: set[str] = set()  # Groq models whose daily allowance ran out
        if mock:
            pass
        elif config.PROVIDER in ("grok", "groq"):
            import openai  # xAI and Groq both serve OpenAI-compatible APIs

            key_name, base_url = {"grok": ("XAI_API_KEY", config.XAI_BASE_URL),
                                  "groq": ("GROQ_API_KEY", config.GROQ_BASE_URL)}[config.PROVIDER]
            key = os.getenv(key_name)
            if not key:
                raise SystemExit(f"FARM_PROVIDER is {config.PROVIDER} but {key_name} is not set (add it to .env).")
            # Groq rate limits are handled in _ask_groq, which can wait longer than the SDK's retries.
            retries = 1 if config.PROVIDER == "groq" else 4
            self.client = openai.OpenAI(api_key=key, base_url=base_url, timeout=900, max_retries=retries)
        else:
            self.client = anthropic.Anthropic(max_retries=4)

    def ask(self, role: str, system: str, prompt: str, *, schema: dict | None = None,
            web: dict | None = None) -> Reply:
        """Run one agent turn. `schema` returns parsed JSON in reply.data; `web` enables web search/fetch."""
        self.meter.check()
        if self.mock:
            return _mock_reply(schema, web)
        if config.PROVIDER == "groq":
            return self._ask_groq(role, system, prompt, schema, web)
        if config.PROVIDER == "grok":
            return self._ask_grok(role, system, prompt, schema, web)
        return self._ask_claude(role, system, prompt, schema, web)

    def _ask_groq(self, role, system, prompt, schema, web) -> Reply:
        """Call Groq, moving to the next model on the account when one model's daily allowance runs out."""
        preferred = config.model_for(role)
        # Only the gpt-oss models can browse; any of them can return structured JSON.
        backups = config.GROQ_WEB_MODELS if web else config.GROQ_WEB_MODELS + config.GROQ_TEXT_MODELS
        candidates = [preferred] + [m for m in backups if m != preferred]
        for model in candidates:
            if model in self._exhausted:
                continue
            try:
                resp = self._groq_call(role, model, system, prompt, schema, web)
                break
            except QuotaExhausted:
                self._exhausted.add(model)
                self.log(f"  {role}: daily allowance for {model} is used up, switching model")
        else:
            raise QuotaExhausted("Groq's free daily allowance is used up on every model")

        message = resp.choices[0].message
        extra = message.model_extra or {}
        text, sources, searches = _groq_sources(message.content or "", extra.get("executed_tools") or [])
        usage = resp.usage
        self.meter.record(model, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0,
                          searches)
        reply = Reply(text=text.strip(), sources=sources)
        if schema:
            if resp.choices[0].finish_reason == "length":
                raise RuntimeError(f"{role} output was cut off")
            reply.data = json.loads(reply.text)
        return reply

    def _groq_call(self, role, model, system, prompt, schema, web):
        import openai

        params = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if model.startswith("openai/gpt-oss"):
            # The free tier leaves little room per request, and long reasoning crowds out the answer,
            # so Groq runs at medium at most (low while browsing, which reads many pages).
            effort = config.effort_for(role)
            params["reasoning_effort"] = "low" if web or effort == "low" else "medium"
        if schema:
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": role, "schema": schema, "strict": True}}
            params["messages"][1]["content"] += ("\n\nBe concise: at most 5 items in any list and one or two "
                                                 "sentences per text field.")
        if web:
            params["tools"] = [{"type": "browser_search"}]
            params["tool_choice"] = "required"

        waited = 0.0
        json_failures = 0
        while True:
            try:
                resp = self.client.chat.completions.create(**params)
                if not (schema and resp.choices[0].finish_reason == "length"):
                    return resp
                problem = "ran out of room"
            except openai.BadRequestError as exc:
                if not schema or "json_validate_failed" not in str(exc):
                    raise
                problem = "returned invalid JSON"
            except openai.RateLimitError as exc:
                message = str(exc)
                if "per day" in message or "(TPD)" in message or "(RPD)" in message:
                    raise QuotaExhausted(f"Groq's free daily limit for {model} is used up") from exc
                wait = _retry_after(exc.response.headers)
                if waited + wait > config.MAX_RATE_WAIT:
                    raise QuotaExhausted(f"Groq rate limit for {model} did not clear") from exc
                self.log(f"  {role}: rate limited on {model}, waiting {wait:.0f}s")
                time.sleep(wait)
                waited += wait
                continue
            except openai.APIStatusError as exc:
                if exc.status_code == 413:
                    raise RuntimeError(f"{role} prompt is too large for Groq's free tier; "
                                       "try fewer --questions") from exc
                raise
            # Open models sometimes miss the schema or run long. Retry, then ask more simply:
            # less reasoning, and finally plain JSON mode with the schema written into the prompt.
            if json_failures >= 3:
                raise RuntimeError(f"{role}: {model} {problem} after 3 retries")
            json_failures += 1
            self.log(f"  {role}: {model} {problem}, retrying ({json_failures}/3)")
            if "reasoning_effort" in params and (json_failures >= 2 or problem == "ran out of room"):
                params["reasoning_effort"] = "low"
            if json_failures == 3:
                params["response_format"] = {"type": "json_object"}
                params["messages"][0]["content"] = (f"{system}\n\nReply with only a JSON object matching this "
                                                    f"JSON schema:\n{json.dumps(schema)}")

    def _ask_grok(self, role, system, prompt, schema, web) -> Reply:
        model = config.model_for(role)
        params = {
            "model": model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if schema:
            params["text"] = {"format": {"type": "json_schema", "name": role, "schema": schema, "strict": True}}
        if web:
            params["tools"] = [{"type": "web_search"}]
        resp = self.client.responses.create(**params)

        sources: dict[str, dict] = {}
        searches = 0
        for item in resp.output:
            if item.type == "web_search_call":
                searches += 1
            elif item.type == "message":
                for part in item.content:
                    for note in getattr(part, "annotations", None) or []:
                        url = getattr(note, "url", None)
                        if url:
                            sources.setdefault(url, {"url": url, "title": getattr(note, "title", "") or "",
                                                     "cited": True})
        # xAI also lists every source it used in a top-level `citations` field.
        for cite in (resp.model_extra or {}).get("citations") or []:
            url = cite if isinstance(cite, str) else cite.get("url")
            if url:
                sources.setdefault(url, {"url": url, "title": "" if isinstance(cite, str) else cite.get("title", "")})

        usage = resp.usage
        self.meter.record(model, usage.input_tokens if usage else 0, usage.output_tokens if usage else 0, searches)
        if resp.status == "incomplete":
            reason = getattr(resp.incomplete_details, "reason", "unknown")
            if schema:
                raise RuntimeError(f"{role} output was incomplete ({reason})")
        reply = Reply(text=resp.output_text.strip(), sources=list(sources.values()))
        if schema:
            reply.data = json.loads(reply.text)
        return reply

    def _ask_claude(self, role, system, prompt, schema, web) -> Reply:
        params = {
            "model": config.model_for(role),
            "max_tokens": 32000 if web else 16000,
            "system": system,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": config.effort_for(role)},
            "betas": [config.FALLBACK_BETA],
            "fallbacks": "default",
        }
        if schema:
            params["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        if web:
            params["tools"] = web_tools(**web)

        messages = [{"role": "user", "content": prompt}]
        parts: list[str] = []
        sources: dict[str, dict] = {}
        for _ in range(config.MAX_CONTINUATIONS + 1):
            with self.client.beta.messages.stream(messages=messages, **params) as stream:
                msg = stream.get_final_message()
            usage = msg.usage
            server = getattr(usage, "server_tool_use", None)
            self.meter.record(
                msg.model,
                usage.input_tokens + (usage.cache_creation_input_tokens or 0) * 1.25
                + (usage.cache_read_input_tokens or 0) * 0.1,
                usage.output_tokens,
                (server.web_search_requests or 0) if server else 0,
            )
            if msg.stop_reason == "refusal":
                details = msg.stop_details
                reason = f"{details.category}: {details.explanation}" if details else "no details"
                raise RefusalError(f"{role} request was declined ({reason})")
            _collect(msg.content, parts, sources)
            if msg.stop_reason != "pause_turn":
                break
            # A long server-tool turn paused; send it back so the model continues where it stopped.
            messages.append({"role": "assistant", "content": msg.content})

        reply = Reply(text="".join(parts).strip(), sources=list(sources.values()))
        if schema:
            if msg.stop_reason == "max_tokens":
                raise RuntimeError(f"{role} output was cut off at max_tokens")
            reply.data = json.loads(reply.text)
        return reply


def _collect(content, parts: list[str], sources: dict[str, dict]) -> None:
    """Gather answer text plus every web source the model saw or cited."""
    for block in content:
        if block.type == "text":
            parts.append(block.text)
            for cite in getattr(block, "citations", None) or []:
                url = getattr(cite, "url", None)
                if url:
                    entry = sources.setdefault(url, {"url": url, "title": getattr(cite, "title", "") or ""})
                    entry["cited"] = True
        elif block.type == "web_search_tool_result":
            if isinstance(block.content, list):  # an error comes back as a single object instead
                for result in block.content:
                    sources.setdefault(result.url, {"url": result.url, "title": result.title or ""})
        elif block.type == "web_fetch_tool_result":
            url = getattr(block.content, "url", None)
            if url:
                sources.setdefault(url, {"url": url, "title": ""})["opened"] = True


def _retry_after(headers) -> float:
    """Seconds to wait before retrying a Groq rate-limited request."""
    if headers.get("retry-after"):
        try:
            return float(headers["retry-after"]) + 1
        except ValueError:
            pass
    reset = headers.get("x-ratelimit-reset-tokens") or ""  # e.g. "1m26.4s" or "5.399s"
    match = re.fullmatch(r"(?:(\d+)m)?([\d.]+)s", reset)
    if match:
        return int(match.group(1) or 0) * 60 + float(match.group(2)) + 1
    return 30.0


_GROQ_CITE = re.compile(r"【(\d+)†[^】]*】")


def _groq_sources(content: str, tools: list[dict]) -> tuple[str, list[dict], int]:
    """Turn Groq's browser citations (【4†L75-L84】 = page opened by tool call 4) into plain URLs."""
    sources: dict[str, dict] = {}
    pages: dict[int, str] = {}
    searches = 0
    for tool in tools:
        results = tool.get("search_results") or {}
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except ValueError:
                results = {}
        items = [r for r in results.get("results") or [] if r.get("url")]
        if tool.get("type") == "browser_search":
            searches += 1
            for r in items:
                sources.setdefault(r["url"], {"url": r["url"], "title": r.get("title") or ""})
        elif items:  # browser.open: one page
            url = items[0]["url"]
            pages[int(tool.get("index", -1))] = url
            sources.setdefault(url, {"url": url, "title": ""})["opened"] = True

    def to_url(match: re.Match) -> str:
        url = pages.get(int(match.group(1)))
        if not url:
            return ""
        sources[url]["cited"] = True
        return f" ({url})"

    return _GROQ_CITE.sub(to_url, content), list(sources.values()), searches


# --- mock mode: exercises the whole pipeline offline, without an API key -------------------------

_counter = itertools.count(1)


def _mock_value(schema: dict, name: str):
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {key: _mock_value(sub, key) for key, sub in schema["properties"].items()}
    if kind == "array":
        return [_mock_value(schema["items"], name) for _ in range(2)]
    if kind == "number":
        return 0.5
    if kind == "integer":
        return 1
    if kind == "boolean":
        return False
    if name in ("source_url", "url") or name.endswith("_url"):
        return f"https://example.com/mock/{next(_counter)}"
    return f"mock {name} {next(_counter)}"


def _mock_reply(schema: dict | None, web: dict | None) -> Reply:
    if schema:
        return Reply(data=_mock_value(schema, "root"))
    sources = [{"url": f"https://example.com/mock/{next(_counter)}", "title": "Mock source"}] if web else []
    return Reply(text="# Mock output\n\nThis text was produced in mock mode.", sources=sources)
