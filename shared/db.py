"""
SQLite persistence for AxiomLoop.

Three tables:
  • notes           — working scratchpad for the in-progress research run.
                      Cleared by researcher.py at the start of each run.
  • research_runs   — archive of every completed research run (query +
                      synthesis + verdict + timing).
  • archived_notes  — per-run snapshot of all notes_add records, keyed
                      back to research_runs by research_id.

mcp_server.py uses the `notes` table directly (the agent's tools work
against it). researcher.py calls `archive_run()` after the loop ends to
copy the working notes into archived_notes and append a new
research_runs row. app.py reads research_runs / archived_notes for the
history view.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "notes.db"


# ── Connection & schema ──────────────────────────────────────────────────────


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source_url TEXT NOT NULL, "
        "subject TEXT NOT NULL, "
        "predicate TEXT NOT NULL, "
        "object TEXT NOT NULL, "
        "quote TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS research_runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "query TEXT NOT NULL, "
        "synthesis TEXT, "
        "verdict_json TEXT, "
        "provider TEXT, "
        "model TEXT, "
        "wall_clock_s REAL, "
        "note_count INTEGER NOT NULL DEFAULT 0, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS archived_notes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "research_id INTEGER NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE, "
        "source_url TEXT NOT NULL, "
        "subject TEXT NOT NULL, "
        "predicate TEXT NOT NULL, "
        "object TEXT NOT NULL, "
        "quote TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_archived_notes_research_id "
        "ON archived_notes(research_id)"
    )
    conn.commit()


# ── Working notes (used by mcp_server.py) ────────────────────────────────────


def clear_working_notes() -> None:
    """Wipe the working notes scratchpad. Called at the start of each run."""
    with connect() as c:
        c.execute("DELETE FROM notes")
        c.commit()


def list_working_notes() -> list[dict[str, Any]]:
    """Read the working notes scratchpad directly (bypasses MCP)."""
    with connect() as c:
        rows = c.execute(
            "SELECT id, source_url, subject, predicate, object, quote, created_at "
            "FROM notes ORDER BY id ASC"
        ).fetchall()
    return [
        {
            "id": r[0],
            "source_url": r[1],
            "subject": r[2],
            "predicate": r[3],
            "object": r[4],
            "quote": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def group_working_notes() -> list[dict[str, Any]]:
    """Bucket working notes by (subject, predicate); classify single/agreement/contradiction."""
    notes = list_working_notes()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for n in notes:
        key = (n["subject"].strip().lower(), n["predicate"].strip().lower())
        buckets.setdefault(key, []).append(n)

    out: list[dict[str, Any]] = []
    for entries in buckets.values():
        objects = {e["object"].strip().lower() for e in entries}
        if len(entries) == 1:
            status = "single"
        elif len(objects) == 1:
            status = "agreement"
        else:
            status = "contradiction"
        out.append({
            "subject": entries[0]["subject"],
            "predicate": entries[0]["predicate"],
            "status": status,
            "entries": entries,
        })
    order = {"contradiction": 0, "agreement": 1, "single": 2}
    out.sort(key=lambda g: (order[g["status"]], -len(g["entries"])))
    return out


# ── Run archival (used by researcher.py) ─────────────────────────────────────


def archive_run(
    *,
    query: str,
    synthesis: str,
    verdict: dict | None,
    provider: str | None,
    model: str | None,
    wall_clock_s: float | None,
) -> int:
    """Snapshot the current working notes into a new research_runs row.

    Returns the new research run id.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    verdict_json = json.dumps(verdict) if verdict is not None else None

    with connect() as c:
        notes = c.execute(
            "SELECT source_url, subject, predicate, object, quote, created_at "
            "FROM notes ORDER BY id ASC"
        ).fetchall()

        cur = c.execute(
            "INSERT INTO research_runs "
            "(query, synthesis, verdict_json, provider, model, wall_clock_s, note_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (query, synthesis, verdict_json, provider, model, wall_clock_s, len(notes), now),
        )
        research_id = cur.lastrowid

        if notes:
            c.executemany(
                "INSERT INTO archived_notes "
                "(research_id, source_url, subject, predicate, object, quote, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(research_id, *n) for n in notes],
            )
        c.commit()
    return research_id


# ── History queries (used by app.py) ─────────────────────────────────────────


def count_runs() -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM research_runs").fetchone()[0]


def list_runs(*, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
    """Return runs newest-first with summary fields only."""
    with connect() as c:
        rows = c.execute(
            "SELECT id, query, provider, model, wall_clock_s, note_count, created_at "
            "FROM research_runs ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [
        {
            "id": r[0],
            "query": r[1],
            "provider": r[2],
            "model": r[3],
            "wall_clock_s": r[4],
            "note_count": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def delete_run(research_id: int) -> bool:
    """Delete a research run and its archived notes (cascade via FK).

    Returns True if a row was deleted.
    """
    with connect() as c:
        cur = c.execute("DELETE FROM research_runs WHERE id = ?", (research_id,))
        c.commit()
        return cur.rowcount > 0


def get_run(research_id: int) -> dict[str, Any] | None:
    """Return one run with its synthesis, verdict, and full note list."""
    with connect() as c:
        row = c.execute(
            "SELECT id, query, synthesis, verdict_json, provider, model, "
            "wall_clock_s, note_count, created_at "
            "FROM research_runs WHERE id = ?",
            (research_id,),
        ).fetchone()
        if not row:
            return None
        notes = c.execute(
            "SELECT id, source_url, subject, predicate, object, quote, created_at "
            "FROM archived_notes WHERE research_id = ? ORDER BY id ASC",
            (research_id,),
        ).fetchall()

    verdict = json.loads(row[3]) if row[3] else None
    return {
        "id": row[0],
        "query": row[1],
        "synthesis": row[2],
        "verdict": verdict,
        "provider": row[4],
        "model": row[5],
        "wall_clock_s": row[6],
        "note_count": row[7],
        "created_at": row[8],
        "notes": [
            {
                "id": n[0],
                "source_url": n[1],
                "subject": n[2],
                "predicate": n[3],
                "object": n[4],
                "quote": n[5],
                "created_at": n[6],
            }
            for n in notes
        ],
    }
