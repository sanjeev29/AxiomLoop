# AxiomLoop — Deep Evidence Researcher

[![Watch the demo on YouTube](https://img.youtube.com/vi/tuks7Ay7bOs/maxresdefault.jpg)](https://youtu.be/tuks7Ay7bOs)

> ▶ [Watch the 2-minute walkthrough on YouTube](https://youtu.be/tuks7Ay7bOs)

Deep Evidence Researcher turns a single question into a structured investigation. You type a question; the agent searches arXiv and Google Scholar for relevant papers, opens the promising ones to read their actual content, and as it goes it saves every concrete claim it finds into a small SQLite notebook — recording not just the claim but its subject, the relation, the value, the verbatim quote, and the source URL. Because every note is structured the same way, the system can automatically group claims that talk about the same thing and flag where sources agree, disagree, or stand alone. When the agent has enough evidence, it writes a short synthesis citing the URLs, and then a separate verifier pass re-reads the notebook to produce a typed report — agreements, disagreements with each side's sources, and a confidence score. The whole loop runs through a multi-provider LLM gateway so you can pick which models (Gemini, Groq, etc.) are allowed to do the thinking, and a Streamlit UI streams every search, page fetch, and note in real time so you can see *how* the answer was built, not just what it is.

## What's inside

- `mcp_server.py` — MCP tools: `search_arxiv`, `search_google_scholar`, `fetch_page`, `notes_add`, `notes_list`, `notes_grouped`
- `researcher.py` — Native tool-use agent loop + verifier turn (typed `VerifierReport`)
- `app.py` — Streamlit UI with live trace, claim groups, and verdict
- `llm_gatewayV2/` — FastAPI gateway in front of 7 LLM providers (own venv) — authored by [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/), vendored here with permission

## Run

**Requirements:** Python 3.14+, `uv`, API keys in `.env` (Gemini / Groq / OpenRouter / etc.)

1. Start the LLM gateway (separate shell):

   ```bash
   cd llm_gatewayV2 && ./run.sh        # serves :8100
   ```

2. CLI:

   ```bash
   uv run python researcher.py "Do LLMs reason or pattern-match?"
   ```

3. Streamlit UI:

   ```bash
   uv run streamlit run app.py
   ```

Notes accumulate in `notes.db` alongside the server.

## Credits

`llm_gatewayV2/` was authored by [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) and vendored into this repo with permission. All other code in this repository is original to AxiomLoop.
