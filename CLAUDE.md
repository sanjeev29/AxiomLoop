# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**AxiomLoop — Deep Evidence Researcher.** An agent that turns a single question into a structured investigation: it searches arXiv & Google Scholar, scrapes promising pages with trafilatura, records structured `(subject, predicate, object, quote)` claims into a SQLite notebook, synthesizes a cited answer, then runs a separate **typed verifier turn** that groups agreements and contradictions across the recorded notes. Past runs are persisted and browsable in the UI.

Native tool-use runs through **LLM Gateway V2** for capability-aware multi-provider routing, prompt caching, and structured-output (`response_format`) support. A Streamlit UI streams the trace turn-by-turn and shows the synthesis + notes once the loop completes.

Requires Python ≥ 3.14. Managed with `uv` (see `uv.lock`, `pyproject.toml`).

## Commands

Start the LLM gateway in a separate shell — it must be reachable on `http://localhost:8100`:

```bash
cd llm_gatewayV2 && ./run.sh        # creates its own .venv, runs on :8100
```

The gateway reads API keys from `AxiomLoop/.env` (`GEMINI_API_KEY`, `GROQ_API_KEY`, `NVIDIA_API_KEY`, `CEREBRAS_API_KEY`, `OPEN_ROUTER_API_KEY`, `GITHUB_ACCESS_TOKEN`, `OLLAMA_MODEL`) plus `LLM_ORDER`. See `.env-template` for the full list.

CLI run:

```bash
uv run python researcher.py "Do LLMs reason or pattern-match?"
```

Streamlit UI:

```bash
uv run streamlit run app.py
```

Sanity-check the MCP server alone:

```bash
uv run python mcp_server.py
```

Gateway provider matrix test (from inside `llm_gatewayV2/`):

```bash
./.venv/bin/python tests/test_all_providers.py
```

## Architecture

Five-layer split, each with one job.

### 1. `mcp_server.py` — FastMCP server (stdio subprocess)

Six tools, all return Pydantic models from `shared/models.py`. Slow tools are wrapped in `_with_timeout(...)` (worker thread + `concurrent.futures` timeout) so a hung scrape never breaks the JSON-RPC stream.

- `search_arxiv(query, max_results, category)` → `list[Paper]` (timeout 20s)
- `search_google_scholar(query, max_results)` → `list[Paper]` via `scholarly` (timeout 25s; may CAPTCHA — surfaces an error the agent falls back from)
- `fetch_page(url)` → `PageContent` via `trafilatura` (timeout 20s)
- `notes_add(source_url, subject, predicate, object, quote)` → `Note` — **five-field structured claim** (the (subject, predicate, object) triple is the key idea; quote is the verbatim snippet)
- `notes_list()` → `list[Note]` — every working note from the current run
- `notes_grouped()` → `list[ClaimGroup]` — buckets notes by `(subject, predicate)`, classifies each group `single` / `agreement` / `contradiction`, sorts contradictions first

All DB access goes through `shared/db.py` (`from shared.db import connect as _db`).

### 2. `researcher.py` — The agent loop

`research(query, *, enabled_providers=None, all_providers=None, max_turns=5, on_event=None) -> ResearchResult` is the importable entrypoint. Running the file directly gives a CLI.

Shape:

- **Wipes the working `notes` table at start** (`clear_working_notes()`).
- Reshapes MCP tool definitions into the gateway's canonical `ToolDef` envelope via `mcp_tool_to_v2()` — that one function is the entire MCP↔gateway bridge.
- Native tool-use with `cache_system=True`, `reasoning="off"` (executor stays cheap). Parallel tool calls in a single turn dispatched concurrently via `asyncio.TaskGroup`.
- **No client-side timeout on `session.call_tool`** — cancelling an in-flight MCP call corrupts the JSON-RPC stream. Timeouts live server-side instead.
- **Per-turn corrective nudge guard.** If the model returns text without tool calls and either (a) no tool has been used yet or (b) it fetched a page but didn't record ≥2 notes, the loop sends one corrective user message and retries. Capped to one nudge per run so weak models don't loop on it.
- Each LLM call and each tool call is emitted as a `TraceEvent` to the `on_event` callback **as it happens** — that's what lets the UI render the loop live.
- After the synthesis turn, **post-run reads come straight from SQLite via `list_working_notes()` / `group_working_notes()`** — not via MCP. Newer FastMCP returns typed results in `structuredContent` rather than `content[0].text`, so the MCP roundtrip silently returned empty before this switch.
- A separate **verifier turn** runs `_run_verifier(notes, query, chat)` against the gateway with `response_format={type:"json_schema", schema: VerifierReport}` and `reasoning="medium"` — typed `VerifierReport` with `agreements`, `disagreements`, `confidence`, `remark`.
- `archive_run(...)` snapshots query + synthesis + verdict + notes into `research_runs` + `archived_notes` and returns the new `research_id`.

**Provider routing — `make_chat(enabled_providers, all_providers)`** (no gateway changes required):
- All enabled (or empty/None): `provider=None` → gateway's full capability-aware failover.
- Exactly one enabled: pinned to that provider.
- Strict subset enabled: client-side mini-failover tries each enabled provider in order on `httpx` error.

### 3. `app.py` — Streamlit UI

- Three top-level **tabs**: `📊 Result` (synthesis + notes + verifier), `📋 Trace` (turn-by-turn with tool-call icons + collapsible details), `🗂 History` (paginated past runs, 5/page, with per-run delete + two-step confirmation + relative timestamps via `_relative_time()`).
- Provider list comes from `GET /v1/providers` + `/v1/capabilities` merged (one shows the universal name set, the other shows live capability flags). Rendered as `st.pills` (multi-select chips). A header button opens a `@st.dialog` modal with the configured/unconfigured matrix.
- Live `st.status` pill at the top of the Result tab updates per event: `"Turn 2 · gemini/gemini-2.5-flash (412ms)"` → `"Turn 2 · fetch_page · https://arxiv.org/..."` → `"Verifying evidence…"` → `"Done · 3 notes · search_arxiv×2, fetch_page×1 · 18.2s"`.
- Empty-state placeholders render in Result and Trace tabs when no run has happened yet in the session.
- Custom CSS at the top: dark theme, gradient title, Outfit font, rounded buttons/expanders, animated primary-button hover.

### 4. `shared/db.py` — SQLite persistence layer

WAL-mode SQLite at `notes.db` (gitignored). Three tables:

- **`notes`** — the agent's working scratchpad. Wiped at start of each run by `clear_working_notes()`.
- **`research_runs`** — every completed run: `id, query, synthesis, verdict_json, provider, model, wall_clock_s, note_count, created_at`.
- **`archived_notes`** — per-run snapshot of all notes the agent recorded; FK to `research_runs(id)` with `ON DELETE CASCADE`. Indexed on `research_id`.

Public API:
- `connect()` — opens, enables WAL + foreign keys, ensures schema.
- `clear_working_notes()`, `list_working_notes()`, `group_working_notes()` — working table.
- `archive_run(...)` — snapshot a run, returns new id.
- `count_runs()`, `list_runs(limit, offset)`, `get_run(id)`, `delete_run(id)` — used by the History view.

### 5. `llm_gatewayV2/` — Self-contained FastAPI gateway

Seven providers (Ollama, Gemini, NVIDIA NIM, Groq, Cerebras, OpenRouter, GitHub Models). Its own `.venv` and `requirements.txt`; the parent `pyproject.toml` does **not** depend on it. The agent talks to it over HTTP via `from client import LLM` (path-injected in `researcher.py`).

Submodules:
- `main.py` — FastAPI routes; server-side JSON-schema validation of `response_format` outputs with one corrective retry.
- `providers.py` — per-provider adapters; `_translate_tools`, `_translate_messages`, `_apply_response_format`, `_apply_reasoning` per dialect.
- `router.py` — RPM/RPD/cooldowns + **capability-aware** `pick()`: failover skips providers that lack a requested capability. Explicit `provider=` bypasses capability gating.
- `cache.py` — Gemini explicit SHA-256-keyed cache (5 min TTL).
- `schemas.py`, `db.py` (gateway's own SQLite log at `gateway_v2.db`), `client.py`.

## Cross-cutting concepts to know before editing

- **`shared/models.py`** is the single source of truth for cross-layer Pydantic models: `Paper`, `WebSearch`, `PageContent`, `Note`, `ClaimGroup`, `Position`, `Disagreement`, `VerifierReport`, `ResearchFinding`, `ResearchState`. Extend these rather than redefining schemas in agents or tools.
- **Working notes vs archived notes.** `notes` table is *scratch for the in-progress run* — it's wiped at the start of every `research()` call. Past runs live in `research_runs` + `archived_notes`, not in `notes`. Anything reading "past data" must read the archive tables.
- **Post-run reads bypass MCP.** Don't call `notes_list` / `notes_grouped` over MCP after the loop — read SQLite directly via `list_working_notes()` / `group_working_notes()`. FastMCP's newer typed-result shape (`structuredContent`) silently broke the JSON parse.
- **Tool timeouts live server-side**, inside each tool body via `_with_timeout(...)`. Never wrap `session.call_tool` in `asyncio.wait_for` — cancellation corrupts the MCP JSON-RPC stream.
- **`provider_meta`** on each tool call is opaque provider-specific state (currently Gemini's `thoughtSignature`) that must be echoed back unchanged on the next assistant turn — without it Gemini 3.x returns HTTP 400.
- **`tool_call_dialect`** in gateway responses is `native`, `prompted_fallback` (Ollama on non-tool-capable models — JSON regexed from prose), or `none`.
- **`reasoning_applied: false` is not an error.** When the current model doesn't support a reasoning knob, the gateway logs a no-op and returns 200. Routing only fails over on actual capability mismatches.
- **MCP server tools stay deterministic.** Extraction / claim-shaping / verification belong agent-side (gateway `response_format` for typed output), not inside tools.
- **System prompt structure** (`SYSTEM_PROMPT` in `researcher.py`) follows the 9-criteria prompt-eval rubric: explicit reasoning (`[SEARCH]/[READ]/[RECORD]/[SYNTH]` tags before each tool call), worked example for `notes_add`, mandatory `notes_add ≥ 2` before synthesis, explicit fallback ladder, `"I cannot answer"` escape. Aim is to keep weak free-tier models on rails without bloating to 50+ lines.

## What this repo deliberately does NOT do

- The MCP server doesn't call any LLM — extraction/synthesis is the agent's job.
- The agent doesn't modify `llm_gatewayV2/` (provider toggle subset failover is implemented client-side in `make_chat`, not via a new gateway endpoint).
- Notes are never reconstructed from the LLM's prose — every cited URL must come from a real `fetch_page` response. The verifier reads notes from the DB, not from chat history.
