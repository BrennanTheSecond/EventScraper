# EventScraper (VT Event Reminders)

Scrapes Virginia Tech event/notice pages and GobblerConnect, runs the content
through an LLM pipeline to extract and rank events, writes a local HTML report,
and emails high-interest matches via Gmail SMTP.

The pipeline uses up to three model roles, and each can be a **different
provider** or the **same provider**:

- **Worker** — extracts the event list from the scraped content.
- **Judge 1** — reviews the worker's output for completeness/correctness.
- **Judge 2** — a second, independent reviewer (optional).

Supported providers: **Ollama** (local/free), **Gemini**, **Claude**
(Anthropic), and **OpenAI**.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `config.json`/`.env` to taste (see below), then run:

```bash
python vt_events_reminder.py            # normal run
python vt_events_reminder.py --dry-run  # run everything, send no email
python vt_events_reminder.py --config other/config.json  # use another config
```

The `CONFIG_PATH` env var and the `--config` flag both point at a different
config file. Secrets are read from a `.env` file next to the config (or from
`DOTENV_PATH`).

---

## Configuration

All **non-secret** settings live in `config.json`. All **secrets** (Gmail app
password, model API keys) live in `.env` (git-ignored).

### Models — `models` section

Each of `worker`, `judge1`, `judge2` is a dict:

```jsonc
"worker": {
  "provider": "ollama",          // "ollama" | "gemini" | "claude" | "openai"
  "url": "http://localhost:11434", // provider endpoint
  "api_key_env": "OLLAMA_API_KEY", // env var in .env holding the API key
  "model": "gemma4:31b"            // model name sent to the provider
}
```

- **`provider`** selects the SDK/request path.
- **`url`** is the base endpoint for that provider (you fill it in, see below).
- **`api_key_env`** names the variable in `.env` that holds the key for this
  model. **Ollama ignores it entirely** — local Ollama needs no key.
- **`model`** is the model identifier.

### Stage behavior — `stages` section

```jsonc
"stages": {
  "use_same_model_for_all": false,   // true => judge1/judge2 reuse the worker model
  "num_judges": 2,                    // 0, 1, or 2 reviewers
  "judge_thinking": {                 // Ollama only: per-model reasoning on/off
    "muse-glimmer:30b": false,
    "qwen3.8:27b": false
  }
}
```

- **`use_same_model_for_all`** — when `true`, you only need to configure
  `models.worker`; the judges use it too. Set `false` to give each stage its
  own provider/model.
- **`num_judges`** — `0` runs the worker only (fastest, no review), `1` runs a
  single judge, `2` runs two independent judges. Defaults to `2`.

### Scraping — `scraping` section

```jsonc
"scraping": {
  "mode": "builtin",   // "builtin" | "web_search"
  ...
}
```

- **`builtin`** — the script fetches each URL in `sources.json` with the
  built-in `requests` + BeautifulSoup parsers and produces structured blocks.
  Use this for **local (Ollama)** workers, which have no built-in web search.
- **`web_search`** — the script hands the list of source URLs to the worker
  model, which uses **its own web-search/browsing tools** to look them up
  itself. Use this for **cloud providers** (Gemini, OpenAI, Claude) that expose
  a model-side search tool. In this mode the deterministic completeness check
  and block counting are skipped (there is no local content to count).

> A good rule of thumb: if the worker provider is Ollama, use `builtin`; if the
> worker is a paid/cloud provider with web-search, use `web_search`.

### Ollama tuning — `ollama` section

```jsonc
"ollama": {
  "num_ctx": 32768,   // context window (tokens)
  "num_gpu": 0,       // -1=all, 0=CPU-only, N=use N GPU layers
  "num_thread": 10,   // inference threads
  "model_unload_grace_seconds": 600
}
```

- `num_ctx`, `num_gpu`, and `num_thread` only apply to Ollama.
- `model_unload_grace_seconds` is the sleep before swapping between different
  Ollama models so the previous model is evicted from RAM. **0 seconds is ideal
  for paid/cloud providers** (no local process to unload); **~600 seconds
  (10 min) is good for Ollama** unloading large local models.

### Email — `email` section

```jsonc
"email": {
  "sender_email": "you@gmail.com",
  "receiver_email": "you@vt.edu",
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 465,
  "send_hour_start": 8,
  "send_hour_end": 24
}
```

The digest is only emailed inside this window; runs finishing earlier hold until
`send_hour_start`. Set `PASSWORD` (a Gmail app password) in `.env`.

### Report — `report` section

```jsonc
"report": {
  "html_file_path": "/abs/path/vt_events_all.html",
  "server_ip": "100.88.84.86",  // IP where the report is served
  "server_port": 6060
}
```

`server_ip` / `server_port` build the URL (e.g. `http://100.88.84.86:6060/
vt_events_all.html`) that is included in the emailed digest so you can open the
full report.

---

## Interests file

`interests.json` (path set via `files.interests`) lists the "high interest"
criteria that determine what lands in the email:

```json
{ "criteria": [ "Offers free food or free items", "...more..." ] }
```

These are rendered verbatim into both the worker and judge prompts. Edit this
one list to change what counts as high-interest.

## Sources file

`sources.json` (path set via `files.sources`) lists the sites the scraper
fetches (or that a web-search worker is told to browse):

```json
{
  "sources": [
    { "id": "events_vt", "name": "Virginia Tech Events",
      "url": "https://events.vt.edu/", "type": "events_vt" }
  ]
}
```

In `builtin` mode each entry's `type` maps to a parser:
`events_vt`, `career_vt`, `gobblerconnect`, or `news`. Unknown types are
skipped. In `web_search` mode only the `name` and `url` are used (the model
browses them itself).

---

## Providers

Defaults are Ollama on this machine (`gemma4:31b` worker, `muse-glimmer:30b`
and `qwen3.8:27b` judges). To use a cloud provider, set the appropriate
`provider`/`url`/`model`, and create the matching `api_key_env` key in `.env`.

### Ollama (local, free)

```jsonc
"worker": { "provider": "ollama", "url": "http://localhost:11434",
            "api_key_env": "OLLAMA_API_KEY", "model": "gemma4:31b" }
```

Requires `ollama serve` running and the models pulled (`ollama pull <model>`).
The API key is ignored. `scraping.mode` should be `builtin`.

### Gemini (Google AI)

```jsonc
"worker": { "provider": "gemini", "url": "https://generativelanguage.googleapis.com",
            "api_key_env": "GEMINI_API_KEY", "model": "gemini-2.0-flash" }
```

Set `GEMINI_API_KEY` in `.env`. Gemini exposes a Google-search tool, so
`scraping.mode: "web_search"` lets it look up the sources itself.

### OpenAI

```jsonc
"worker": { "provider": "openai", "url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY", "model": "gpt-4o" }
```

Set `OPENAI_API_KEY` in `.env`. Use `scraping.mode: "web_search"` for models
that have web browsing; otherwise provide content another way.

### Claude (Anthropic)

```jsonc
"worker": { "provider": "claude", "url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY", "model": "claude-3-5-sonnet" }
```

Set `ANTHROPIC_API_KEY` in `.env`. `provider` may be `"claude"` or
`"anthropic"` — both are accepted.

Any of the four providers can be used for the worker *and/or* the judges, and
they may be mixed (e.g. an Ollama worker with a Gemini judge).

---

## Secrets (`.env`)

```
PASSWORD=your-gmail-app-password
OLLAMA_API_KEY=
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

`.env` is git-ignored. Never commit real credentials.

## License

MIT — see `LICENSE`.
