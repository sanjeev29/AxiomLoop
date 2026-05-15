"""
Deep Evidence researcher.

Native tool-use loop against the MCP server defined in mcp_server.py
(search_arxiv, search_google_scholar, fetch_page, notes_add, notes_list).
Uses LLM Gateway V2 for native tool dispatch, parallel tool calls via
asyncio.TaskGroup, and prompt caching of the system block.

Importable: `await research(query, on_event=cb)` streams TraceEvent objects
to a callback so a UI (Streamlit) can render the loop live.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "llm_gatewayV2"))
from client import LLM  # type: ignore[import-not-found]  # noqa: E402  # resolved at runtime via sys.path

from shared.db import (  # noqa: E402
    archive_run,
    clear_working_notes,
    group_working_notes,
    list_working_notes,
)
from shared.models import VerifierReport  # noqa: E402


# ── Schemas ──────────────────────────────────────────────────────────────────


class ToolDef(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    kind: Literal["llm_call", "tool_call", "summary", "verdict", "error"]
    turn: int
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    text: str | None = None
    payload: dict | None = None


class ResearchResult(BaseModel):
    query: str
    summary: str
    events: list[TraceEvent] = Field(default_factory=list)
    notes: list[dict] = Field(default_factory=list)
    groups: list[dict] = Field(default_factory=list)
    verdict: VerifierReport | None = None
    started_at: float = Field(default_factory=time.time)
    wall_clock_s: float | None = None
    research_id: int | None = None


EventCallback = Optional[Callable[[TraceEvent], Awaitable[None] | None]]


# ── System prompt ────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a research agent. Answer the user's question using a small toolbelt — aim to finish in 3–4 turns.

Think before each step. At the start of each turn, write ONE short reasoning line tagged with the type of step — [SEARCH], [READ], or [SYNTH] — then issue your tool call(s) on the same turn. Example:
  [SEARCH] Need recent benchmarks on LLM reasoning — searching arXiv and Scholar in parallel.

Required flow:
  1. [SEARCH] First turn: call search_arxiv AND search_google_scholar together (parallel).
  2. [READ] Next: fetch_page on the 1–2 most promising URLs from the searches (parallel is fine).
  3. [RECORD] After reading, you MUST call notes_add at least TWICE — one note per concrete claim from the pages you just read. Each notes_add takes: (source_url, subject, predicate, object, quote). Parallel calls are fine. Subject = entity, predicate = relation, object = value/property, quote = verbatim snippet. Keep the same subject/predicate wording when two sources are talking about the same thing so the verifier can group them.
     Example call:
       notes_add(source_url="https://arxiv.org/abs/2206.07682", subject="large language models", predicate="exhibit", object="emergent abilities at scale", quote="capability X appears only above ~10^22 FLOPs.")
  4. [SYNTH] Before the final answer, silently self-check: (a) have I actually read ≥1 source via fetch_page? (b) have I recorded ≥2 notes via notes_add? (c) is every URL I'm about to cite one I actually read? If any check fails, do the missing tool call instead of synthesizing.

Final answer format — plain text, 3–5 sentences, with inline URL citations to pages you read (e.g. "X holds when … (https://…)."). The final message must contain NO tool calls.

Rules:
  • Prefer parallel tool calls when the calls are independent (two searches, or two fetches).
  • Never fabricate a URL, title, or quote — only cite what fetch_page actually returned.
  • If a tool result is empty or starts with 'ERROR:', do NOT retry the same call. Fall back: scholar fails → use arxiv; arxiv empty → broaden the query (fewer keywords); fetch_page empty → try a different URL.
  • If you cannot make progress after two search attempts with zero usable results, reply starting with "I cannot answer"."""


# ── Helpers ──────────────────────────────────────────────────────────────────


def mcp_tool_to_v2(t) -> dict:
    return ToolDef(
        name=t.name,
        description=t.description or "",
        input_schema=t.inputSchema or {"type": "object", "properties": {}},
    ).model_dump()


async def dispatch_tool_calls(session, tool_calls: list[dict]) -> list[dict]:
    """Dispatch each tool call in parallel.

    Timeouts live inside the tool implementations on the server side
    (see mcp_server.py:_with_timeout) — cancelling an in-flight
    session.call_tool from the client breaks the MCP stdio JSON-RPC stream.
    """
    async def run_one(tc: dict) -> dict:
        name = tc["name"]
        try:
            result = await session.call_tool(name, tc.get("arguments") or {})
            text = result.content[0].text if result.content else ""
        except Exception as e:
            text = f"ERROR: {type(e).__name__}: {e}"
        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "tool_name": name,
            "content": text,
        }

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(run_one(tc)) for tc in tool_calls]
    return [t.result() for t in tasks]


async def _emit(cb: EventCallback, event: TraceEvent) -> None:
    if cb is None:
        return
    res = cb(event)
    if asyncio.iscoroutine(res):
        await res


VERIFIER_SYSTEM = (
    "You are an evidence verifier. You receive a structured table of research "
    "notes (subject, predicate, object, quote, source_url) recorded by an "
    "agent. Return a VerifierReport that:\n"
    "  - lists agreements: statements multiple sources support (one sentence each)\n"
    "  - lists disagreements: topics where sources take different positions, "
    "with each position's stance and the source URLs that hold it\n"
    "  - confidence: 0-1 overall confidence that the notes support a coherent answer "
    "(low if too few notes, mixed evidence, or weak sources)\n"
    "  - remark: one-line overall comment\n"
    "Be terse. Do not invent claims not present in the notes."
)


def make_chat(
    enabled_providers: list[str] | None = None,
    all_providers: list[str] | None = None,
) -> Callable[..., dict]:
    """Return a chat callable honouring an enabled-provider subset.

    Strategy (no changes to the gateway — we only choose what to pass):
      - enabled is None, empty, or covers every known provider: let the
        gateway auto-route via its capability-aware failover (provider=None).
      - enabled has exactly one entry: pin to that provider.
      - enabled is a strict subset: try each in order on httpx errors —
        client-side mini-failover restricted to the user's selection.
    """
    llm = LLM()
    enabled = list(enabled_providers or [])
    known = set(all_providers or [])

    def call(**kwargs) -> dict:
        kwargs.pop("provider", None)
        if not enabled or (known and set(enabled) >= known):
            return llm.chat(provider=None, **kwargs)
        if len(enabled) == 1:
            return llm.chat(provider=enabled[0], **kwargs)
        last_err: Exception | None = None
        for p in enabled:
            try:
                return llm.chat(provider=p, **kwargs)
            except Exception as e:
                last_err = e
                continue
        raise last_err if last_err else RuntimeError("no enabled providers responded")

    return call


def _run_verifier(
    notes: list[dict],
    query: str,
    chat: Callable[..., dict],
) -> VerifierReport | None:
    """Make a separate gateway call to produce a typed VerifierReport."""
    if not notes:
        return None
    schema = VerifierReport.model_json_schema()
    reply = chat(
        prompt=(
            f"Research question: {query}\n\n"
            f"Notes ({len(notes)} total):\n{json.dumps(notes, indent=2)}"
        ),
        system=VERIFIER_SYSTEM,
        cache_system=True,
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "VerifierReport",
            "strict": True,
        },
        reasoning="medium",
        temperature=0,
        max_tokens=1024,
    )
    if reply.get("parsed"):
        try:
            return VerifierReport.model_validate(reply["parsed"])
        except Exception:
            pass
    return VerifierReport(
        agreements=[],
        disagreements=[],
        confidence=0.0,
        remark="structured-output not honoured by provider; verifier skipped",
    )


# ── Main entrypoint ──────────────────────────────────────────────────────────


async def research(
    query: str,
    *,
    enabled_providers: list[str] | None = None,
    all_providers: list[str] | None = None,
    max_turns: int = 8,
    on_event: EventCallback = None,
) -> ResearchResult:
    """Run the research loop. Streams trace events to on_event as they happen.

    `enabled_providers` honours the user's per-provider on/off selection from
    the UI; the gateway itself is not modified. See `make_chat` for routing
    semantics.
    """
    started_at = time.time()
    events: list[TraceEvent] = []
    chat = make_chat(enabled_providers, all_providers)

    # Wipe the working notes scratchpad so this run starts clean — past runs
    # live in archived_notes / research_runs.
    clear_working_notes()

    async def record(ev: TraceEvent) -> None:
        events.append(ev)
        await _emit(on_event, ev)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = [mcp_tool_to_v2(t) for t in mcp_tools]

            messages: list[dict] = [{"role": "user", "content": query}]
            summary_text = ""

            for turn in range(1, max_turns + 1):
                reply = chat(
                    messages=messages,
                    system=SYSTEM_PROMPT,
                    cache_system=True,
                    tools=tools,
                    tool_choice="auto",
                    reasoning="off",
                    temperature=0,
                    max_tokens=2048,
                )

                await record(TraceEvent(
                    kind="llm_call",
                    turn=turn,
                    provider=reply.get("provider"),
                    model=reply.get("model"),
                    latency_ms=reply.get("latency_ms"),
                    input_tokens=reply.get("input_tokens"),
                    output_tokens=reply.get("output_tokens"),
                    text=reply.get("text"),
                    payload={"tool_calls": reply.get("tool_calls", [])},
                ))

                tool_calls = reply.get("tool_calls") or []
                text = (reply.get("text") or "").strip()
                if not tool_calls:
                    used_any_tool = any(e.kind == "tool_call" for e in events)
                    notes_added = sum(
                        1 for e in events
                        if e.kind == "tool_call" and e.tool_name == "notes_add"
                    )
                    fetched_any = any(
                        e.kind == "tool_call" and e.tool_name == "fetch_page"
                        for e in events
                    )

                    # Allow at most one nudge so we don't loop forever on weak models.
                    nudges_used = sum(
                        1 for m in messages
                        if m.get("role") == "user"
                        and "Please make progress" in str(m.get("content", ""))
                    )

                    nudge_reason: str | None = None
                    if not used_any_tool and len(text) < 80:
                        nudge_reason = (
                            "Please make progress by calling a tool now "
                            "(search_arxiv, search_google_scholar, fetch_page). "
                            "If you truly cannot proceed, say so starting with "
                            "'I cannot answer'."
                        )
                    elif fetched_any and notes_added < 2:
                        nudge_reason = (
                            "Please make progress: before synthesizing you must "
                            "call notes_add at least twice (one per concrete "
                            f"claim) — currently {notes_added}. Use the pages "
                            "you already fetched and call notes_add in parallel."
                        )

                    if nudge_reason and turn < max_turns and nudges_used < 1:
                        messages.append({"role": "assistant", "content": text})
                        messages.append({"role": "user", "content": nudge_reason})
                        continue

                    summary_text = text
                    break

                messages.append({
                    "role": "assistant",
                    "content": reply.get("text", "") or "",
                    "tool_calls": tool_calls,
                })

                results = await dispatch_tool_calls(session, tool_calls)
                for tc, r in zip(tool_calls, results):
                    await record(TraceEvent(
                        kind="tool_call",
                        turn=turn,
                        tool_name=tc["name"],
                        tool_args=tc.get("arguments"),
                        tool_result=r["content"],
                    ))
                messages.extend(results)
            else:
                summary_text = f"(stopped after {max_turns} turns without final synthesis)"

            # Read working notes + groups directly from SQLite. We deliberately
            # skip the MCP round-trip here because newer FastMCP versions return
            # typed results under `structuredContent` rather than as a JSON
            # string in `content[0].text`, which made the previous reader
            # silently fall back to an empty list.
            notes = list_working_notes()
            groups = group_working_notes()

            await record(TraceEvent(kind="summary", turn=0, text=summary_text))

            # Separate verifier turn — typed VerifierReport via response_format.
            verdict = _run_verifier(notes, query, chat)
            if verdict is not None:
                await record(TraceEvent(
                    kind="verdict",
                    turn=0,
                    payload=verdict.model_dump(),
                ))

            wall_clock_s = round(time.time() - started_at, 2)

            # Persist the run for the history view (synthesis + all notes).
            last_llm = next(
                (e for e in reversed(events) if e.kind == "llm_call"), None
            )
            research_id = archive_run(
                query=query,
                synthesis=summary_text,
                verdict=verdict.model_dump() if verdict else None,
                provider=last_llm.provider if last_llm else None,
                model=last_llm.model if last_llm else None,
                wall_clock_s=wall_clock_s,
            )

            return ResearchResult(
                query=query,
                summary=summary_text,
                events=events,
                notes=notes,
                groups=groups,
                verdict=verdict,
                started_at=started_at,
                wall_clock_s=wall_clock_s,
                research_id=research_id,
            )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="+", help="research question")
    p.add_argument(
        "--provider",
        default=None,
        help="Pin to a single provider (e.g. 'groq', 'gemini'). Omit for auto-routing.",
    )
    args = p.parse_args()

    def printer(ev: TraceEvent) -> None:
        if ev.kind == "llm_call":
            print(f"\n[turn {ev.turn}] {ev.provider}/{ev.model} "
                  f"({ev.latency_ms}ms, in={ev.input_tokens} out={ev.output_tokens})")
            if ev.text:
                print(f"  text: {ev.text!r}")
        elif ev.kind == "tool_call":
            args_str = json.dumps(ev.tool_args or {})[:120]
            result_str = (ev.tool_result or "")[:200].replace("\n", " ")
            print(f"  ↪ {ev.tool_name}({args_str}) → {result_str}")
        elif ev.kind == "summary":
            print(f"\n=== SUMMARY ===\n{ev.text}")
        elif ev.kind == "verdict":
            p = ev.payload or {}
            print(f"\n=== VERDICT (confidence={p.get('confidence')}) ===")
            print(f"  remark: {p.get('remark')}")
            for a in p.get("agreements") or []:
                print(f"  ✓ {a}")
            for d in p.get("disagreements") or []:
                print(f"  ✗ {d.get('topic')}")
                for pos in d.get("positions") or []:
                    print(f"      - {pos.get('stance')}  [{', '.join(pos.get('sources') or [])}]")

    enabled = [args.provider] if args.provider else None
    result = asyncio.run(research(
        " ".join(args.query),
        enabled_providers=enabled,
        on_event=printer,
    ))
    print(f"\n[notes: {len(result.notes)}]  [groups: {len(result.groups)}]  [wall: {result.wall_clock_s}s]")
