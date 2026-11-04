from __future__ import annotations

from dataclasses import dataclass, field

from .models import Chunk, Ticket
from .text import clean, windows


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1500
    overlap: int = 200
    max_body_chunks: int = 8
    index_comments: bool = True
    max_comment_chunks: int = 20
    min_comment_chars: int = 40
    skip_comment_authors: frozenset[str] = field(default_factory=frozenset)


def chunk_ticket(ticket: Ticket, config: ChunkingConfig) -> list[Chunk]:
    title = ticket.title.strip()
    body_windows = windows(clean(ticket.description), config.max_chars, config.overlap)
    chunks = [
        Chunk("body", i, _with_title(title, window))
        for i, window in enumerate(body_windows[: config.max_body_chunks])
    ]
    if not chunks:
        chunks.append(Chunk("body", 0, title))
    if not config.index_comments:
        return chunks
    return chunks + _comment_chunks(ticket, title, config)


def _comment_chunks(ticket: Ticket, title: str, config: ChunkingConfig) -> list[Chunk]:
    skip = {author.lower() for author in config.skip_comment_authors}
    result: list[Chunk] = []
    for comment in ticket.comments:
        if comment.author.lower() in skip:
            continue
        body = clean(comment.body)
        if len(body) < config.min_comment_chars:
            continue
        for window in windows(body, config.max_chars, config.overlap):
            if len(result) >= config.max_comment_chunks:
                return result
            result.append(Chunk("comment", len(result), _with_title(title, window)))
    return result


def _with_title(title: str, text: str) -> str:
    return f"{title}\n\n{text}" if title else text
