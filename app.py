"""
Streamlit UI for the Deep Evidence researcher.

Run:
    streamlit run app.py

Requires the LLM Gateway V2 to be running on localhost:8100
(cd llm_gatewayV2 && ./run.sh).
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import streamlit as st

from datetime import datetime, timezone

from researcher import ResearchResult, TraceEvent, research
from shared.db import count_runs, delete_run, get_run, list_runs


def _relative_time(iso_str: str | None) -> str:
    """Turn an ISO-8601 UTC timestamp into a friendly relative string."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 10:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr ago"
    days = hours // 24
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if dt.year == now.year:
        return dt.strftime("%b %d")
    return dt.strftime("%b %d, %Y")

GATEWAY_URL = os.environ.get("LLM_GATEWAY_V2_URL", "http://localhost:8100")

st.set_page_config(
    page_title="Deep Evidence Researcher",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
      
      html, body, [class*="css"]  {
          font-family: 'Outfit', sans-serif !important;
      }
      
      #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
      .block-container { 
          padding-top: 2rem; 
          padding-bottom: 4rem; 
          max-width: 1200px; 
      }
      
      h1 { 
          font-weight: 700; 
          letter-spacing: -0.02em; 
          background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
      }
      
      .subtle { 
          color: var(--text-color); 
          opacity: 0.7; 
          font-size: 1.05rem; 
          font-weight: 300;
          line-height: 1.5;
      }
      
      /* Buttons */
      .stButton > button { 
          border-radius: 12px; 
          font-weight: 600; 
          transition: all 0.3s ease;
          border: none;
      }
      .stButton > button[data-testid="baseButton-primary"] {
          background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
          color: white;
          box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
      }
      .stButton > button[data-testid="baseButton-primary"]:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(118, 75, 162, 0.6);
      }
      .stButton > button[data-testid="baseButton-secondary"] {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
      }
      .stButton > button[data-testid="baseButton-secondary"]:hover {
          background: rgba(255, 255, 255, 0.1);
          transform: translateY(-2px);
      }
      
      /* Inputs */
      .stTextInput > div > div > input {
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.1);
          padding: 0.75rem 1rem;
          font-size: 1.1rem;
          transition: all 0.3s ease;
      }
      .stTextInput > div > div > input:focus {
          border-color: #667EEA;
          box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3);
      }
      
      /* Metrics */
      [data-testid="stMetricValue"] { 
          font-size: 2rem; 
          font-weight: 700;
          color: #4ECDC4;
      }
      [data-testid="stMetricLabel"] {
          font-size: 1rem;
          opacity: 0.8;
      }
      
      /* Cards and Expanders */
      .stStatus { 
          border-radius: 12px; 
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.05);
      }
      div[data-testid="stExpander"] { 
          border-radius: 12px; 
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.05);
          backdrop-filter: blur(10px);
          overflow: hidden;
      }
      div[data-testid="stExpander"] > summary {
          padding: 1rem;
          font-weight: 500;
      }
      div[data-testid="stExpander"] > summary:hover {
          background: rgba(255,255,255,0.05);
      }
      [data-testid="stVerticalBlockBorderWrapper"] { 
          border-radius: 12px; 
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(255,255,255,0.05);
          transition: all 0.2s;
      }
      [data-testid="stVerticalBlockBorderWrapper"]:hover {
          border-color: rgba(102, 126, 234, 0.4);
          background: rgba(255,255,255,0.04);
      }
      
      /* Pills */
      .st-emotion-cache-1g8lwn6 { 
          border-radius: 20px;
      }
      
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
          gap: 2rem;
          background-color: transparent;
      }
      .stTabs [data-baseweb="tab"] {
          height: 3.5rem;
          border-radius: 8px 8px 0 0;
          font-size: 1.1rem;
          font-weight: 500;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data ─────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def fetch_providers() -> dict:
    """Merge /v1/providers (universal list) with /v1/capabilities (live detail)."""
    try:
        prov = httpx.get(f"{GATEWAY_URL}/v1/providers", timeout=5.0).json()
        caps = httpx.get(f"{GATEWAY_URL}/v1/capabilities", timeout=5.0).json()
    except Exception as e:
        return {"_error": str(e)}

    active = set(prov.get("providers") or [])
    merged: dict[str, dict] = {}
    for name in sorted((prov.get("limits") or {}).keys()):
        merged[name] = {**(caps.get(name) or {}), "active": name in active}
    return merged


# ── Header ───────────────────────────────────────────────────────────────────


providers = fetch_providers()
gateway_ok = "_error" not in providers
all_providers = list(providers.keys()) if gateway_ok else []
active_providers = [n for n, info in providers.items() if info.get("active")] if gateway_ok else []


@st.dialog("LLM providers", width="large")
def providers_dialog() -> None:
    if not gateway_ok:
        st.error(f"Gateway unreachable at {GATEWAY_URL}.")
        st.caption("Start it with `cd llm_gatewayV2 && ./run.sh`.")
        return
    st.caption(
        f"{len(active_providers)} of {len(all_providers)} known providers "
        "are configured on the gateway. Add API keys to `.env` to enable more."
    )
    cols = st.columns(2)
    for i, name in enumerate(all_providers):
        info = providers[name]
        active = info.get("active")
        model = info.get("model") or "—"
        flags = [k for k in ("tools", "reasoning", "structured", "caching") if info.get(k)]
        badge = "🟢" if active else "⚪"
        with cols[i % 2]:
            st.markdown(
                f"{badge} **{name}**  ·  `{model}`  \n"
                f'<span class="subtle">{" · ".join(flags) if flags else "—"}</span>',
                unsafe_allow_html=True,
            )


# ── Header ───────────────────────────────────────────────────────────────────


head_l, head_r = st.columns([5, 1])
with head_l:
    st.title("🔎 Deep Evidence Researcher")
    st.markdown(
        '<p class="subtle">'
        "Searches arXiv & Google Scholar, scrapes promising pages, records structured "
        "claims, and produces a cross-referenced answer with a typed verifier."
        "</p>",
        unsafe_allow_html=True,
    )
with head_r:
    st.write("")
    btn_label = (
        f"⚙  {len(active_providers)}/{len(all_providers)} providers"
        if gateway_ok else "⚙  gateway down"
    )
    if st.button(btn_label, use_container_width=True, type="secondary"):
        providers_dialog()


# ── Controls ─────────────────────────────────────────────────────────────────

query = st.text_input(
    "Research question",
    value="Do large language models reason or are they sophisticated pattern matchers?",
    label_visibility="collapsed",
    placeholder="Ask anything — e.g. 'Do LLMs reason, or pattern-match?'",
)

ctl1, ctl2 = st.columns([3, 1])

with ctl1:
    if not gateway_ok:
        st.error(f"Gateway unreachable at {GATEWAY_URL} — start it with `cd llm_gatewayV2 && ./run.sh`")
        enabled_providers: list[str] = []
    elif not active_providers:
        st.warning(
            "No providers configured on the gateway. Add API keys to `.env` "
            "(GEMINI_API_KEY, GROQ_API_KEY, etc.) and restart the gateway."
        )
        enabled_providers = []
    else:
        enabled_providers = st.pills(
            "Providers",
            options=active_providers,
            default=active_providers,
            selection_mode="multi",
            label_visibility="collapsed",
            format_func=lambda n: n,
        )

with ctl2:
    max_turns = st.slider(
        "Max turns",
        2, 12, 5,
        label_visibility="visible",
        help=(
            "How many steps the agent is allowed to take before it must give "
            "you an answer. A step is one round of thinking — the agent can "
            "do several things in one step (search multiple sources, read "
            "pages, save notes). Most questions finish in 3–4 steps. Raise "
            "for a more thorough answer, lower for a quicker one."
        ),
    )

run = st.button(
    "Research",
    type="primary",
    use_container_width=True,
    disabled=not query.strip() or (gateway_ok and not enabled_providers),
)

# ── Run ──────────────────────────────────────────────────────────────────────


st.divider()

# Tabs — always rendered. When idle, Result/Trace show empty-state hints.
result_tab, trace_tab, history_tab = st.tabs(["📊  Result", "📋  Trace", "🗂  History"])

if not run:
    with result_tab:
        st.markdown(
            "<div style='text-align:center; padding: 4rem 1rem; opacity: 0.55;'>"
            "<div style='font-size: 3rem;'>📊</div>"
            "<div style='font-size: 1.1rem; font-weight: 500; margin-top: 0.5rem;'>"
            "No research yet</div>"
            "<div style='font-size: 0.95rem; margin-top: 0.5rem;'>"
            "Type a question above and hit <b>Research</b> — the synthesis, notes, "
            "and verifier report will land here."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with trace_tab:
        st.markdown(
            "<div style='text-align:center; padding: 4rem 1rem; opacity: 0.55;'>"
            "<div style='font-size: 3rem;'>📋</div>"
            "<div style='font-size: 1.1rem; font-weight: 500; margin-top: 0.5rem;'>"
            "No trace yet</div>"
            "<div style='font-size: 0.95rem; margin-top: 0.5rem;'>"
            "Each search, page fetch, and note the agent records will stream "
            "here turn-by-turn during a run."
            "</div></div>",
            unsafe_allow_html=True,
        )

if run:
    # Live status pill anchored to the Result tab, mirrors current activity.
    with result_tab:
        live_status = st.status(
            "Spinning up the agent…",
            expanded=True,
            state="running",
        )
        st.caption("Open the **Trace** tab above to watch each step.")
        # Body container fills with synthesis + notes + verifier once the run completes.
        result_body = st.container()

    with trace_tab:
        trace_container = st.container()

    turn_blocks: dict[int, object] = {}
    tool_counts: dict[str, int] = {}

    def _tool_summary(name: str, args: dict | None) -> str:
        a = args or {}
        if name in ("search_arxiv", "search_google_scholar"):
            q = (a.get("query") or "").strip()
            return f"{name} · “{q[:60]}{'…' if len(q) > 60 else ''}”"
        if name == "fetch_page":
            return f"fetch_page · {a.get('url', '')[:80]}"
        if name == "notes_add":
            subj = a.get("subject", "?")
            pred = a.get("predicate", "?")
            obj = (a.get("object") or "")[:40]
            return f"notes_add · {subj} {pred} {obj}"
        return name

    def on_event(ev: TraceEvent) -> None:
        # ── Top status: "what's happening RIGHT NOW" ─────────────────
        if ev.kind == "llm_call":
            live_status.update(
                label=(
                    f"Turn {ev.turn} · thinking with "
                    f"{ev.provider or '—'} / {ev.model or '—'} "
                    f"({ev.latency_ms or 0} ms)"
                ),
                state="running",
            )
        elif ev.kind == "tool_call":
            tool_counts[ev.tool_name or "?"] = tool_counts.get(ev.tool_name or "?", 0) + 1
            summary = _tool_summary(ev.tool_name or "", ev.tool_args)
            live_status.update(
                label=f"Turn {ev.turn} · {summary}",
                state="running",
            )
        elif ev.kind == "summary":
            live_status.update(label="Verifying evidence…", state="running")
        elif ev.kind == "verdict":
            live_status.update(label="Wrapping up…", state="running")
        elif ev.kind == "error":
            live_status.update(label=f"Error: {ev.text}", state="error")

        # ── Trace history below ──────────────────────────────────────
        with trace_container:
            if ev.kind == "llm_call":
                title = (
                    f"Turn {ev.turn}  ·  {ev.provider or '—'}/{ev.model or '—'}  "
                    f"·  {ev.latency_ms or 0} ms  ·  "
                    f"in {ev.input_tokens or 0} / out {ev.output_tokens or 0}"
                )
                block = st.status(title, expanded=True, state="complete")
                turn_blocks[ev.turn] = block
                if ev.text:
                    with block:
                        st.markdown(f"_{ev.text}_")
            elif ev.kind == "tool_call":
                block = turn_blocks.get(ev.turn)
                target = block if block is not None else trace_container
                with target:
                    icon = {
                        "search_arxiv": "🔎",
                        "search_google_scholar": "🎓",
                        "fetch_page": "📄",
                        "notes_add": "📝",
                        "notes_list": "📚",
                        "notes_grouped": "🗂️",
                    }.get(ev.tool_name or "", "🛠")
                    summary = _tool_summary(ev.tool_name or "", ev.tool_args)
                    is_error = (ev.tool_result or "").startswith("ERROR:")
                    if is_error:
                        st.markdown(f"{icon} **`{ev.tool_name}`** — :red[{ev.tool_result}]")
                    else:
                        st.markdown(f"{icon} **`{ev.tool_name}`** — {summary}")
                    with st.expander("details", expanded=False):
                        if ev.tool_args:
                            st.code(json.dumps(ev.tool_args, indent=2), language="json")
                        result_preview = (ev.tool_result or "")[:1500]
                        if result_preview:
                            st.code(result_preview, language="text")
            elif ev.kind == "summary":
                st.success("Synthesis complete")
            elif ev.kind == "verdict":
                p = ev.payload or {}
                conf = p.get("confidence") or 0
                st.info(
                    f"Verifier  ·  confidence {conf:.2f}  ·  "
                    f"{len(p.get('agreements') or [])} agreements  ·  "
                    f"{len(p.get('disagreements') or [])} disagreements"
                )
            elif ev.kind == "error":
                st.error(ev.text or "unknown error")

    try:
        result: ResearchResult = asyncio.run(
            research(
                query.strip(),
                enabled_providers=enabled_providers or None,
                all_providers=all_providers or None,
                max_turns=max_turns,
                on_event=on_event,
            )
        )
        tc_summary = ", ".join(f"{k}×{v}" for k, v in tool_counts.items()) or "no tools used"
        live_status.update(
            label=f"Done · {len(result.notes)} notes · {tc_summary} · {result.wall_clock_s}s",
            state="complete",
            expanded=False,
        )

        with result_body:
            st.markdown("### Synthesis")
            st.markdown(result.summary or "_(no synthesis returned)_")

            st.markdown(f"### Notes  ·  {len(result.notes)}")
            if result.notes:
                for note in result.notes:
                    n = note if isinstance(note, dict) else (note.model_dump() if hasattr(note, "model_dump") else dict(note))
                    with st.container(border=True):
                        st.markdown(f"**{n.get('subject', '')}** {n.get('predicate', '')} **{n.get('object', '')}**")
                        if n.get("quote"):
                            st.markdown(f"> *\"{n.get('quote')}\"*")
                        source = n.get("source_url") or n.get("source") or ""
                        if source:
                            st.caption(f"[Source]({source})")
            else:
                st.warning(
                    "**No notes were recorded.** The prompt requires at least "
                    "two `notes_add` calls before synthesis, but the model "
                    "skipped them. This usually means a weak free-tier model "
                    "or too few turns. Try a stronger provider (Groq, larger "
                    "Gemini) or raise Max turns."
                )

            if result.verdict is not None:
                v = result.verdict
                st.markdown("### Verifier report")
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{v.confidence:.2f}")
                m2.metric("Agreements", len(v.agreements))
                m3.metric("Disagreements", len(v.disagreements))
                if v.remark:
                    st.caption(v.remark)
                if v.agreements:
                    st.markdown("**Agreements**")
                    for a in v.agreements:
                        st.markdown(f"- {a}")
                if v.disagreements:
                    st.markdown("**Disagreements**")
                    for d in v.disagreements:
                        with st.expander(d.topic, expanded=False):
                            for pos in d.positions:
                                srcs = ", ".join(pos.sources) or "_no source_"
                                st.markdown(f"- **{pos.stance}** — {srcs}")

            if result.groups:
                with st.expander("Claim groups (raw)", expanded=False):
                    st.json(result.groups)
    except BaseException as e:
        # Flatten BaseExceptionGroup so the real underlying error is visible.
        def _flatten(exc: BaseException) -> list[BaseException]:
            if isinstance(exc, BaseExceptionGroup):
                out: list[BaseException] = []
                for sub in exc.exceptions:
                    out.extend(_flatten(sub))
                return out
            return [exc]

        leaves = _flatten(e)
        primary = leaves[0] if leaves else e
        live_status.update(
            label=f"Failed: {type(primary).__name__}",
            state="error",
            expanded=True,
        )
        st.error(f"{type(primary).__name__}: {primary}")
        for i, leaf in enumerate(leaves):
            with st.expander(
                f"Sub-exception {i + 1}/{len(leaves)} — {type(leaf).__name__}",
                expanded=(i == 0),
            ):
                st.exception(leaf)


# ── History (past research runs) ─────────────────────────────────────────────


def _render_run_detail(run: dict) -> None:
    """Render a single archived run's synthesis + notes + verdict."""
    st.markdown(f"**Query**  ·  {run['query']}")
    meta_bits = []
    if run.get("provider"):
        meta_bits.append(f"{run['provider']}/{run.get('model') or '—'}")
    if run.get("wall_clock_s") is not None:
        meta_bits.append(f"{run['wall_clock_s']}s")
    if run.get("created_at"):
        meta_bits.append(_relative_time(run["created_at"]))
    if meta_bits:
        st.caption("  ·  ".join(meta_bits))

    st.markdown("**Synthesis**")
    st.markdown(run.get("synthesis") or "_(no synthesis)_")

    notes = run.get("notes") or []
    if notes:
        st.markdown(f"**Notes ({len(notes)})**")
        for note in notes:
            n = note if isinstance(note, dict) else (note.model_dump() if hasattr(note, "model_dump") else dict(note))
            with st.container(border=True):
                st.markdown(f"**{n.get('subject', '')}** {n.get('predicate', '')} **{n.get('object', '')}**")
                if n.get("quote"):
                    st.markdown(f"> *\"{n.get('quote')}\"*")
                source = n.get("source_url") or n.get("source") or ""
                if source:
                    st.caption(f"[Source]({source})")

    v = run.get("verdict") or {}
    if v:
        agreements = v.get("agreements") or []
        disagreements = v.get("disagreements") or []
        st.markdown(
            f"**Verifier**  ·  confidence {v.get('confidence', 0):.2f}  "
            f"·  {len(agreements)} agreements  ·  {len(disagreements)} disagreements"
        )
        if agreements:
            for a in agreements:
                st.markdown(f"- ✓ {a}")
        if disagreements:
            for d in disagreements:
                with st.expander(d.get("topic", "Disagreement"), expanded=False):
                    for pos in d.get("positions") or []:
                        srcs = ", ".join(pos.get("sources") or []) or "_no source_"
                        st.markdown(f"- **{pos.get('stance')}** — {srcs}")


with history_tab:
    PAGE_SIZE = 5
    total_runs = count_runs()

    if total_runs == 0:
        st.caption("No past research yet. Run one above and it'll appear here.")
    else:
        if "history_page" not in st.session_state:
            st.session_state.history_page = 0

        last_page = max(0, (total_runs - 1) // PAGE_SIZE)
        st.session_state.history_page = min(st.session_state.history_page, last_page)

        page = st.session_state.history_page
        offset = page * PAGE_SIZE
        runs = list_runs(limit=PAGE_SIZE, offset=offset)

        for hrun in runs:
            title = hrun["query"][:90] + ("…" if len(hrun["query"]) > 90 else "")
            sub = (
                f"#{hrun['id']}  ·  {_relative_time(hrun.get('created_at'))}  ·  "
                f"{hrun.get('note_count', 0)} notes"
            )
            with st.expander(f"**{title}**\n\n{sub}", expanded=False):
                full = get_run(hrun["id"])
                if full:
                    _render_run_detail(full)

                st.divider()
                confirm_key = f"confirm_delete_{hrun['id']}"
                if st.session_state.get(confirm_key):
                    warn_l, warn_r = st.columns([3, 1])
                    with warn_l:
                        st.warning(f"Delete run #{hrun['id']} permanently?")
                    with warn_r:
                        cl, cr = st.columns(2)
                        if cl.button("Cancel", key=f"cancel_{hrun['id']}", use_container_width=True):
                            st.session_state[confirm_key] = False
                            st.rerun()
                        if cr.button(
                            "Delete",
                            key=f"do_delete_{hrun['id']}",
                            type="primary",
                            use_container_width=True,
                        ):
                            delete_run(hrun["id"])
                            st.session_state[confirm_key] = False
                            st.rerun()
                else:
                    if st.button(
                        "🗑  Delete run",
                        key=f"delete_{hrun['id']}",
                        type="secondary",
                    ):
                        st.session_state[confirm_key] = True
                        st.rerun()

        nav_l, nav_m, nav_r = st.columns([1, 2, 1])
        with nav_l:
            if st.button("← Prev", disabled=(page == 0), use_container_width=True):
                st.session_state.history_page -= 1
                st.rerun()
        with nav_m:
            first = offset + 1
            last = min(offset + PAGE_SIZE, total_runs)
            st.markdown(
                f"<p style='text-align:center; opacity:0.6;'>"
                f"Showing {first}–{last} of {total_runs}  ·  page {page + 1} / {last_page + 1}"
                f"</p>",
                unsafe_allow_html=True,
            )
        with nav_r:
            if st.button("Next →", disabled=(page >= last_page), use_container_width=True):
                st.session_state.history_page += 1
                st.rerun()
