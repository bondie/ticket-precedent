from __future__ import annotations

import re
import unicodedata

_JIRA_MARKUP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\{(code|noformat|quote|panel|color)(:[^}]*)?\}"), " "),
    (re.compile(r"!\S+?\.(png|jpe?g|gif|svg|bmp)(\|[^!]*)?!", re.IGNORECASE), " "),
    (re.compile(r"\[~([^\]]+)\]"), r"@\1"),
    (re.compile(r"\[([^|\]]+)\|[^\]]+\]"), r"\1"),
    (re.compile(r"^h[1-6]\.\s*", re.MULTILINE), ""),
    (re.compile(r"^bq\.\s*", re.MULTILINE), ""),
]
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_WORD = re.compile(r"\w+", re.UNICODE)


def clean(text: str | None) -> str:
    if not text:
        return ""
    for pattern, replacement in _JIRA_MARKUP:
        text = pattern.sub(replacement, text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def windows(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    result: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _soft_break(text, start + max_chars // 2, end)
        piece = text[start:end].strip()
        if piece:
            result.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return result


def _soft_break(text: str, lo: int, hi: int) -> int:
    for separator in ("\n", ". ", " "):
        cut = text.rfind(separator, lo, hi)
        if cut != -1:
            return cut + len(separator)
    return hi


def normalize_token(token: str) -> str:
    decomposed = unicodedata.normalize("NFKD", token.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _WORD.finditer(text):
        token = normalize_token(match.group())
        if len(token) >= 2:
            seen.setdefault(token, None)
    return list(seen)
