"""Fetch a web page as plain text and cut it down to the parts relevant to some claims.

Used by page checking: the farm re-reads a finding's source itself instead of trusting the model's memory of it.
"""
import re
import urllib.request
from html.parser import HTMLParser

MAX_BYTES = 3_000_000
SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "template"}
BLOCK_TAGS = {"p", "div", "li", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre",
              "blockquote", "td", "th", "dd", "dt"}
STOPWORDS = set("the and for with that this from are was were has have had its into than then them they their "
                "which while when where what who will would could should about over under more most such only also "
                "been being can may not but all any each other some very".split())


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def fetch_text(url: str, timeout: float = 20) -> str:
    """Download a page and return its readable text. Raises on network errors or non-HTML content."""
    if not re.match(r"^https?://", url or ""):
        raise ValueError("not a web URL")
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (research-farm source checker)", "Accept": "text/html,text/plain;q=0.9"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        kind = resp.headers.get_content_type()
        if kind not in ("text/html", "text/plain", "application/xhtml+xml"):
            raise ValueError(f"unsupported content type {kind}")
        raw = resp.read(MAX_BYTES)
        charset = resp.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, errors="replace")
    return clean_text(text) if kind == "text/plain" else html_to_text(text)


def html_to_text(html: str) -> str:
    """Readable text of an HTML page: scripts, styles and markup removed, one block per line."""
    parser = _Text()
    parser.feed(html)
    return clean_text("".join(parser.parts))


def clean_text(text: str) -> str:
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOPWORDS}


def relevant_excerpt(text: str, claims: list[str], budget: int, chunk: int = 1200) -> str:
    """Keep the chunks of `text` that share the most words with the claims, in page order, within `budget` chars."""
    if len(text) <= budget:
        return text
    chunks = [text[i:i + chunk] for i in range(0, len(text), chunk)]
    wanted = _words(" ".join(claims))
    scored = sorted(range(len(chunks)), key=lambda i: len(wanted & _words(chunks[i])), reverse=True)
    keep = sorted(scored[: max(1, budget // chunk)])
    return "\n[...]\n".join(chunks[i] for i in keep)
