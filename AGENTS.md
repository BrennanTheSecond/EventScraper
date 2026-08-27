# AGENTS.md — VT Event Reminders

## One-sentence purpose
Scrapes Virginia Tech event/notice pages + GobblerConnect, runs content through an LLM (Ollama, Gemini, Claude, or OpenAI) to extract and rank events, writes a local HTML report, and emails high-interest matches via Gmail SMTP.

## Entrypoint
- `vt_events_reminder.py` — single-file script, run directly: `python vt_events_reminder.py` (add `--dry-run` to skip email).

## Setup
- Python venv at `.venv/` (user creates it: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`) — installs `python-dotenv`, `requests`, `bs4`, `ollama`, `google-generativeai`, `openai`, `anthropic`.
- Config split across several files:
  - `config.json` — non-secret settings: `models` (per-stage provider/url/key-ref/model), `stages` (same-model toggle, `num_judges` 0–2, judge thinking), `scraping` (`mode` builtin|web_search + tunables), `ollama` (ctx/gpu/thread/grace), `email`, `report` (`html_file_path`, `server_ip`, `server_port`), `files` (paths to interests/sources).
  - `interests.json` — the `criteria` list rendered into worker + judge prompts.
  - `sources.json` — the sites to scrape/browse, each with a `type` mapping to a parser.
  - `.env` — secrets only (`PASSWORD`, `OLLAMA_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Git-ignored.
- Model support is provider-agnostic: each stage (worker/judge1/judge2) can be `ollama`, `gemini`, `claude`/`anthropic`, or `openai`. Default is local Ollama on this device: worker `gemma4:31b`, judges `muse-glimmer:30b` and `qwen3.8:27b`.

## Config path override
- Script reads `CONFIG_PATH` env var or `--config` flag (default: `config.json` next to the script). Secrets come from `.env` next to the config, or from `DOTENV_PATH`. `interests.json`/`sources.json` resolve relative to the config file.
- The script exposes loaded values as module globals (`MODEL_SPECS`, `WORKER_SPEC`, `JUDGE1_SPEC`, `JUDGE2_SPEC`, `NUM_JUDGES`, `SCRAPE_MODE`, `SOURCES`, `INTEREST_CRITERIA`, `REPORT_URL`, etc.) so the rest of the file reads them unchanged.

## Scrape modes
- `scraping.mode = "builtin"` — the requests/BeautifulSoup parsers fetch each source in `sources.json`. Use for local (Ollama) workers.
- `scraping.mode = "web_search"` — hand the source URL list to the worker model, which browses them with its own web-search tool. Use for cloud providers (Gemini/OpenAI/Claude). The deterministic completeness check / block count is skipped in this mode.

## Credentials (in `.env` — handle with care)
- `PASSWORD` — Gmail app password (do not commit; `.env` is git-ignored).
- `OLLAMA_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — per-provider keys, referenced by each model's `api_key_env`. Ollama ignores its key.
- `SENDER_EMAIL` / `RECEIVER_EMAIL` — in `config.json` `email` section.
- **Never commit real credentials.** Rotate the Gmail app password — a real one was previously committed in source history.

## Important paths
- Output HTML: `config.json` → `report.html_file_path`.
- Emailed report URL is built from `report.server_ip` / `report.server_port`.
- Config: `config.json`; interests: `interests.json`; sources: `sources.json`.

## Architecture notes
- No tests, no CI, no linting/formatting config. README.md documents provider setup.
- No package structure — single module; config/globals are loaded from `config.json` + `interests.json` + `sources.json` + `.env` at the top.
- GobblerConnect is fetched via CampusGroups JSON web service (`requests`); no Selenium/Chromium needed.
- LLM responses are parsed as JSON; the script strips markdown fences before `json.loads`.
- Prompts ask the LLM to return structured `{"all_events": [...], "filtered_events": [...]}`.
- Providers are dispatched via `call_model(spec, prompt, thinking, label)`; Ollama-specific unload/thread tuning is in `_call_ollama`.

## Common pitfalls
- Script will crash if Ollama is not running locally (`ollama serve` must be active, model pulled) and an Ollama provider is configured.
- Email will fail (and a WARNING prints) if `PASSWORD` is missing from `.env`.
- No error handling if the LLM returns malformed JSON (prints raw response and exits).
- `report.html_file_path` is an absolute path — must match actual file location if moved.
- Cloud providers require their SDK (`openai`, `anthropic`, `google-generativeai`) and a configured `api_key_env` populated in `.env`.
