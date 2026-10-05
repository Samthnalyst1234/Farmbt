"""Farm settings. Every value can be overridden with an environment variable (or a line in .env)."""
import os


def _detect_provider() -> str:
    for provider, key in (("claude", "ANTHROPIC_API_KEY"), ("grok", "XAI_API_KEY"), ("groq", "GROQ_API_KEY")):
        if os.getenv(key):
            return provider
    return "claude"


# Which AI provider runs the agents:
#   claude  Anthropic Claude   (ANTHROPIC_API_KEY)
#   grok    xAI Grok           (XAI_API_KEY)
#   groq    Groq, free tier    (GROQ_API_KEY) - open models, small rate limits
# If unset, the farm uses the first of those keys it finds.
PROVIDER = os.getenv("FARM_PROVIDER") or _detect_provider()

# Model used by every agent unless a role-specific override is set,
# e.g. FARM_MODEL_RESEARCHER=claude-sonnet-5 to run the researchers on a cheaper model.
# Claude role models must support adaptive thinking and effort (Opus 5.x, Sonnet 5, Fable 5.x).
DEFAULT_MODEL = os.getenv("FARM_MODEL", {
    "claude": "claude-opus-5",
    "grok": "grok-4.7",
    "groq": "openai/gpt-oss-120b",
}[PROVIDER])

# On Groq each model has its own daily token allowance, so the web-heavy roles use a second model,
# and when one model's allowance runs out the farm moves to the next one on the same account.
GROQ_ROLE_MODELS = {"researcher": "openai/gpt-oss-20b", "extractor": "openai/gpt-oss-20b",
                    "verifier": "openai/gpt-oss-20b", "prior_art": "openai/gpt-oss-20b"}
GROQ_WEB_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b"]  # can browse
GROQ_TEXT_MODELS = ["qwen/qwen3.8-27b"]  # no browsing; used for the other roles as a last resort

XAI_BASE_URL = "https://api.x.ai/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# How hard each role thinks: low | medium | high | xhigh | max. (Groq uses low | medium | high.)
ROLE_EFFORT = {
    "planner": "high",
    "researcher": "medium",
    "extractor": "low",
    "analyst": "high",
    "gap_hunter": "xhigh",
    "critic": "medium",
    "reporter": "medium",
    "solution_designer": "high",
    "solution_critic": "medium",
    "experiment_planner": "high",
    "verifier": "medium",
    "gap_merger": "medium",
    "prior_art": "medium",
}

# Harness profile: the same agents, sized to what the model and account can handle.
#   lite      small/free models and tight rate limits: few questions, short prompts, one worker
#   standard  mid-size paid models
#   deep      frontier models: more questions, more searching and reading, more gaps solved
# Chosen automatically from the provider and model; override with FARM_PROFILE.
def _detect_profile() -> str:
    if PROVIDER == "groq":
        return "lite"
    if any(tag in DEFAULT_MODEL for tag in ("opus", "fable", "mythos")):
        return "deep"
    return "standard"


PROFILE = os.getenv("FARM_PROFILE") or _detect_profile()
_PROFILES = {
    "lite": {
        "questions": 3, "workers": 1, "findings_per_question": 8, "evidence_chars": 150, "solve": 1,
        "researcher_web": {"searches": 3, "fetches": 2}, "critic_web": {"searches": 5, "fetches": 1},
        "verify_pages": 4, "page_chars": 9000, "gap_votes": 3, "prior_art_web": {"searches": 4, "fetches": 2},
    },
    "standard": {
        "questions": 5, "workers": 4, "findings_per_question": 20, "evidence_chars": 600, "solve": 1,
        "researcher_web": {"searches": 5, "fetches": 3}, "critic_web": {"searches": 8, "fetches": 2},
        "verify_pages": 12, "page_chars": 30000, "gap_votes": 3, "prior_art_web": {"searches": 6, "fetches": 3},
    },
    "deep": {
        "questions": 7, "workers": 4, "findings_per_question": 40, "evidence_chars": 2000, "solve": 2,
        "researcher_web": {"searches": 8, "fetches": 5}, "critic_web": {"searches": 12, "fetches": 4},
        "verify_pages": 30, "page_chars": 80000, "gap_votes": 3, "prior_art_web": {"searches": 10, "fetches": 5},
    },
}
_p = _PROFILES[PROFILE]
FINDINGS_PER_QUESTION = _p["findings_per_question"]
EVIDENCE_CHARS = _p["evidence_chars"]
RESEARCHER_WEB = _p["researcher_web"]  # web limits apply to Claude; Grok and Groq decide for themselves
CRITIC_WEB = _p["critic_web"]
PRIOR_ART_WEB = _p["prior_art_web"]
# Page checking: how many source pages to re-read per run, and how much of each page the verifier sees.
VERIFY_PAGES = int(os.getenv("FARM_VERIFY_PAGES", str(_p["verify_pages"])))
PAGE_CHARS = _p["page_chars"]
# Majority voting: independent gap-hunting passes; a gap must appear in most of them to survive.
GAP_VOTES = int(os.getenv("FARM_GAP_VOTES", str(_p["gap_votes"])))

# Compact prompts: Groq's free tier rejects any request over ~8,000 tokens.
COMPACT = PROFILE == "lite"
MAX_RATE_WAIT = 15 * 60  # seconds to wait out a rate limit before pausing the run

DEFAULT_SOLVE = int(os.getenv("FARM_SOLVE", str(_p["solve"])))  # top gaps the solver stage works on per run
DEFAULT_QUESTIONS = int(os.getenv("FARM_QUESTIONS", str(_p["questions"])))
DEFAULT_WORKERS = int(os.getenv("FARM_WORKERS", str(_p["workers"])))
DEFAULT_MAX_COST = float(os.getenv("FARM_MAX_COST", "5"))  # USD per session; 0 = no limit

DB_PATH = os.getenv("FARM_DB", "farm.db")
REPORTS_DIR = os.getenv("FARM_REPORTS", "reports")

# Refusal fallback: if a request is declined, the API re-runs it on Anthropic's recommended fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_CONTINUATIONS = 5  # resumes of a server-tool turn that stopped with pause_turn

# USD per million tokens (input, output). Used for the spend meter only.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "grok-4.7": (2.0, 6.0),
    "grok-build-0.1": (1.0, 2.0),
    "openai/": (0.0, 0.0),  # Groq free tier
    "qwen/": (0.0, 0.0),
}
WEB_SEARCH_PRICE = {"claude": 10.0 / 1000, "grok": 5.0 / 1000, "groq": 0.0}  # USD per search


def model_for(role: str) -> str:
    override = os.getenv(f"FARM_MODEL_{role.upper()}")
    if override:
        return override
    if PROVIDER == "groq" and not os.getenv("FARM_MODEL"):
        return GROQ_ROLE_MODELS.get(role, DEFAULT_MODEL)
    return DEFAULT_MODEL


def effort_for(role: str) -> str:
    return os.getenv(f"FARM_EFFORT_{role.upper()}", ROLE_EFFORT[role])


def price_for(model: str) -> tuple[float, float]:
    # Longest prefix first so claude-opus-5-5 doesn't match claude-opus-5.
    for name in sorted(PRICES, key=len, reverse=True):
        if model.startswith(name):
            return PRICES[name]
    return PRICES["grok-4.7" if model.startswith("grok") else "claude-opus-5"]
