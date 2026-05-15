"""
MCP server for the Deep Evidence researcher.

Tools:
  - search_arxiv:          arXiv API (free, no key)
  - search_google_scholar: Google Scholar via `scholarly` (free, may CAPTCHA)
  - fetch_page:            HTML URL → readable text (trafilatura)
  - notes_add / notes_list / notes_grouped:
                           SQLite-backed structured evidence notebook
                           (subject, predicate, object, quote per claim;
                           notes_grouped flags agreement vs contradiction)

Run standalone for a sanity check:
    python mcp_server.py
"""

from __future__ import annotations

import concurrent.futures
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _with_timeout(func: Callable[[], T], *, seconds: float, label: str) -> T:
    """Run a blocking callable in a worker thread with a timeout.

    Server-side timeouts keep the MCP JSON-RPC stream intact — cancelling
    a tool call from the client side would leave the protocol half-written.
    On timeout, the worker thread is leaked (Python can't kill it) but the
    main server thread returns control immediately with a TimeoutError that
    FastMCP turns into a clean error response to the agent.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func)
        try:
            return fut.result(timeout=seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{label} timed out after {seconds:.0f}s")

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.db import connect as _db  # noqa: E402
from shared.models import ClaimGroup, Note, PageContent, Paper  # noqa: E402

mcp = FastMCP("AxiomLoop-Research-Server")


@mcp.tool()
def search_arxiv(
    query: str,
    max_results: int = 5,
    category: Optional[str] = None,
) -> list[Paper]:
    """Search arXiv. `category` is an arXiv taxonomy code (e.g. 'cs.LG', 'cs.CL')."""
    def _do() -> list[Paper]:
        import arxiv

        q = f"cat:{category} AND ({query})" if category else query
        client = arxiv.Client(page_size=min(max_results, 50), num_retries=3)
        search = arxiv.Search(
            query=q,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out: list[Paper] = []
        for r in client.results(search):
            out.append(Paper(
                title=r.title.strip(),
                authors=[a.name for a in r.authors],
                year=r.published.year if r.published else None,
                venue="arXiv",
                abstract=(r.summary or "").strip() or None,
                url=r.entry_id,
                pdf_url=r.pdf_url,
                citations=None,
                source="arxiv",
            ))
        return out

    return _with_timeout(_do, seconds=20, label="search_arxiv")


@mcp.tool()
def search_google_scholar(query: str, max_results: int = 5) -> list[Paper]:
    """Search Google Scholar via `scholarly`. May CAPTCHA — bounded by timeout."""
    def _do() -> list[Paper]:
        from scholarly import scholarly

        it = scholarly.search_pubs(query)
        out: list[Paper] = []
        for _ in range(max_results):
            try:
                r = next(it)
            except StopIteration:
                break
            bib = r.get("bib", {}) or {}
            authors = bib.get("author") or []
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(" and ") if a.strip()]
            year = bib.get("pub_year")
            try:
                year = int(year) if year else None
            except (TypeError, ValueError):
                year = None
            out.append(Paper(
                title=(bib.get("title") or "").strip() or "(untitled)",
                authors=authors,
                year=year,
                venue=bib.get("venue") or None,
                abstract=bib.get("abstract") or None,
                url=r.get("pub_url") or r.get("eprint_url") or "",
                pdf_url=r.get("eprint_url") or None,
                citations=r.get("num_citations"),
                source="scholar",
            ))
        return out

    return _with_timeout(_do, seconds=25, label="search_google_scholar")


@mcp.tool()
def fetch_page(url: str) -> PageContent:
    """Fetch a URL and return readable text + metadata (boilerplate stripped)."""
    def _do() -> PageContent:
        import json

        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return PageContent(url=url, text="", title=None)

        result = trafilatura.extract(
            downloaded,
            output_format="json",
            with_metadata=True,
            include_comments=False,
        )
        if not result:
            return PageContent(url=url, text="", title=None)

        data = json.loads(result)
        return PageContent(
            url=url,
            title=data.get("title"),
            text=data.get("text", ""),
            author=data.get("author"),
            date=data.get("date"),
        )

    return _with_timeout(_do, seconds=20, label="fetch_page")


@mcp.tool()
def notes_add(
    source_url: str,
    subject: str,
    predicate: str,
    object: str,
    quote: str | None = None,
) -> Note:
    """Save one structured evidence claim.

    The (subject, predicate, object) triple lets the agent (and the verifier)
    group corroborating claims and detect contradictions. Examples:
      subject="GPT-4", predicate="exhibits", object="emergent reasoning"
      subject="LLMs", predicate="solve", object="novel math problems"
    `quote` is the verbatim snippet from the source that supports the claim.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO notes (source_url, subject, predicate, object, quote, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source_url, subject, predicate, object, quote, now),
        )
        note_id = cur.lastrowid
    return Note(
        id=note_id,
        source_url=source_url,
        subject=subject,
        predicate=predicate,
        object=object,
        quote=quote,
        created_at=now,
    )


@mcp.tool()
def notes_list() -> list[Note]:
    """Return every structured note saved so far, oldest first."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, source_url, subject, predicate, object, quote, created_at "
            "FROM notes ORDER BY id ASC"
        ).fetchall()
    return [
        Note(
            id=r[0], source_url=r[1], subject=r[2], predicate=r[3],
            object=r[4], quote=r[5], created_at=r[6],
        )
        for r in rows
    ]


@mcp.tool()
def notes_grouped() -> list[ClaimGroup]:
    """Bucket notes by (subject, predicate) and flag each group.

    status:
      - "single":        only one source recorded this claim
      - "agreement":     multiple sources, all reporting the same `object`
      - "contradiction": multiple sources reporting different `object` values
    """
    notes = notes_list()
    buckets: dict[tuple[str, str], list[Note]] = {}
    for n in notes:
        key = (n.subject.strip().lower(), n.predicate.strip().lower())
        buckets.setdefault(key, []).append(n)

    groups: list[ClaimGroup] = []
    for (subj, pred), entries in buckets.items():
        objects = {e.object.strip().lower() for e in entries}
        if len(entries) == 1:
            status = "single"
        elif len(objects) == 1:
            status = "agreement"
        else:
            status = "contradiction"
        groups.append(ClaimGroup(
            subject=entries[0].subject,
            predicate=entries[0].predicate,
            status=status,
            entries=entries,
        ))
    # Show contradictions first — they're what the agent should drill into.
    order = {"contradiction": 0, "agreement": 1, "single": 2}
    groups.sort(key=lambda g: (order[g.status], -len(g.entries)))
    return groups


if __name__ == "__main__":
    mcp.run(transport="stdio")
