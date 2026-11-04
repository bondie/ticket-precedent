from ticket_precedent.chunking import ChunkingConfig, chunk_ticket
from ticket_precedent.models import Comment
from ticket_precedent.text import clean, tokens, windows

from .conftest import make_ticket


def test_clean_strips_jira_markup():
    raw = (
        "h3. Steps\n{code:java}\nfoo();\n{code}\n"
        "See [the docs|https://x.y] and !shot.png|thumbnail! [~alice]"
    )
    assert clean(raw) == "Steps\n\nfoo();\n\nSee the docs and @alice"


def test_clean_strips_html_but_keeps_generics_in_code():
    raw = "<p>ahoj Palo,</p><p>pro&scaron;li jsme <strong>feedy</strong></p>Map<String, List<X>> m"
    assert clean(raw) == "ahoj Palo,\n\nprošli jsme feedy\nMap<String, List<X>> m"


def test_windows_cover_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    parts = windows(text, max_chars=500, overlap=100)
    assert len(parts) > 1
    assert all(len(part) <= 500 for part in parts)
    assert parts[0].split()[0] == "word0"
    assert parts[-1].split()[-1] == "word399"
    assert parts[1].split()[0] in parts[0]


def test_tokens_fold_case_and_czech_diacritics():
    assert tokens("Objednávka se NEPROPSALA, objednavka") == ["objednavka", "se", "nepropsala"]


def test_chunks_keep_title_and_skip_noise_comments():
    ticket = make_ticket(
        "D-1",
        "Invoice missing in shop",
        "The ERP sent the invoice but the shop never shows it.",
        comments=["+1", "Root cause: the comparator dropped the payload as unchanged, reset it."],
    )
    ticket.comments.append(
        Comment("Jenkins", None, "Build #42 passed for commit abcdef, all good.")
    )
    chunks = chunk_ticket(ticket, ChunkingConfig(skip_comment_authors=frozenset({"jenkins"})))
    assert [c.kind for c in chunks] == ["body", "comment"]
    assert all(c.text.startswith("Invoice missing in shop") for c in chunks)
    assert "comparator" in chunks[1].text


def test_ticket_without_description_still_has_a_chunk():
    chunks = chunk_ticket(make_ticket("D-2", "Only a title"), ChunkingConfig())
    assert [c.text for c in chunks] == ["Only a title"]


def test_comments_can_be_disabled():
    ticket = make_ticket("D-3", "T", "body text", comments=["a long enough comment " * 3])
    chunks = chunk_ticket(ticket, ChunkingConfig(index_comments=False))
    assert [c.kind for c in chunks] == ["body"]
