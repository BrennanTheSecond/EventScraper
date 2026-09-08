# EventScraper (VT Event Reminders)

Scrapes Virginia Tech event/notice pages and GobblerConnect into structured
records, asks an LLM which of them match your interests, writes a local HTML
report, and emails the matches via Gmail SMTP.

**Extraction is deterministic.** Each source has a dedicated parser, and what
those parsers produce *is* the event list — no model is involved in deciding
which events exist, when they are, or whether the source advertises free food.
That last one is read from the listing's own CSS facets, so it is a fact rather
than an inference.

The model is asked exactly one question, a batch of events at a time: *which of
these interest criteria does this event match?* It answers with a couple of
digits per event.

Up to three reviewers vote on that question, each of which can be a
**different provider** or the same one:

- **Worker** — classifies every event.
- **Judge 1 / Judge 2** — optional independent second and third opinions.

Reviewers vote by **union**: an event is high-interest if any of them says it
matches. Missing a free-food event is the failure that matters; an extra line in
the digest costs nothing.

Supported providers: **Ollama** (local/free), **Gemini**, **Claude**
(Anthropic), and **OpenAI**.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`config.json` is git-ignored because it holds your email addresses and server
IP. Start from the tracked example:

```bash
cp config.example.json config.json   # then edit it (see below)
```

Create `.env` next to it for the secrets, then run:

```bash
python vt_events_reminder.py            # normal run
python vt_events_reminder.py --dry-run  # run everything, send no email
python vt_events_reminder.py --config other/config.json  # use another config
```

The `CONFIG_PATH` env var and the `--config` flag both point at a different
config file. Secrets are read from a `.env` file next to the config (or from
`DOTENV_PATH`).

### Iterating without waiting for a full run

A real run is dominated by model generation and can take hours, which makes it
a terrible way to test a prompt tweak or a change to the report. These three
flags cut the loop to under a second:

```bash
# Scrape once, keep the result.
python vt_events_reminder.py --dry-run --save-scrape scrape.json

# Replay that scrape instead of refetching every source.
python vt_events_reminder.py --dry-run --from-scrape scrape.json

# Replay it AND skip every model call, building the event list straight from
# the scraped blocks. Exercises prompts, merging, the report and the email
# body end to end with no LLM.
python vt_events_reminder.py --no-llm --from-scrape scrape.json
```

`--no-llm` implies `--dry-run`: no email is sent and the report is written
beside the real one as `*_dryrun.html`.

### Tests

```bash
python test_pipeline.py
```

No pytest, no network, no model. Covers the pure functions that turn untrusted
input into decisions: parsing a model's reply, de-duplicating events across
sources, week filtering, day grouping, HTML escaping, the scrape cache
round-trip, and the digest bodies.

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
  "num_judges": 2,                    // 0, 1, or 2 extra reviewers
  "classify_batch_size": 25,          // events per model call
  "classify_detail_limit": 400,       // chars of description sent per event
  "call_max_attempts": 3,             // retries per model call
  "call_backoff_seconds": 5,          // doubles each retry
  "cloud_timeout_seconds": 180,       // per-request timeout (cloud providers)
  "max_tokens": 8192,                 // reply cap (cloud providers)
  "model_thinking": {                 // Ollama only: per-model reasoning on/off
    "muse-glimmer:30b": false,
    "qwen3.8:27b": false
  }
}
```

- **`use_same_model_for_all`** — when `true`, you only need to configure
  `models.worker`; the judges use it too. Set `false` to give each stage its
  own provider/model.
- **`num_judges`** — `0` runs the worker alone (fastest), `1` or `2` add
  independent reviewers whose matches are unioned with the worker's. Defaults
  to `2`. Each judge costs a full pass over every batch, so on a local box this
  is the main quality-versus-time dial.
- **`classify_batch_size`** — events per model call. Larger batches mean fewer
  calls but a longer reply to keep coherent; 25 is a good default. A batch whose
  reply cannot be parsed is skipped, so the blast radius of one bad reply is
  that batch, not the run.
- **`classify_detail_limit`** — how much of each event's description reaches the
  prompt. The interest decision rarely needs more than the first couple of
  sentences, and this is the single biggest lever on prefill cost.
- **`call_max_attempts` / `call_backoff_seconds`** — retry policy for every
  model call, with exponential backoff. Errors that a retry cannot fix (unknown
  model, bad API key, a model that rejects `think`) are detected and not
  retried.
- **`model_thinking`** — Ollama reasoning on/off per model. A model absent from
  this map keeps Ollama's own default. `judge_thinking` is still read as an
  alias.

  You do not need to know which of your models support reasoning. Ollama
  rejects a `think` request outright for a model that has none
  (`400 "does not support thinking"`); the script catches that, **drops
  `think` and runs the model anyway**, and remembers not to ask again this run.
  So setting a model to `true` here is a preference, not an assertion — a model
  that cannot reason simply runs without it instead of failing the batch.

### Scraping — `scraping` section

```jsonc
"scraping": {
  "mode": "builtin",        // the only supported mode
  "include_undated": true,  // keep news notices that carry no date
  ...
}
```

- **`mode`** must be `"builtin"`. The old `"web_search"` mode asked a cloud
  model to browse the sources and report what it found — that is model-driven
  extraction, which this pipeline no longer does at all. The config is rejected
  with an explanation rather than silently ignored.
- **`include_undated`** — `news.vt.edu` notices carry no date in the listing
  markup, so unlike every other source they cannot be week-filtered. Keep them
  (they land in an "Undated notices" group in the report) or set this to `false`
  to drop them.

### Ollama tuning — `ollama` section

```jsonc
"ollama": {
  "num_ctx": 32768,   // context window (tokens)
  "num_gpu": 0,       // -1=all, 0=CPU-only, N=use N GPU layers
  "num_thread": 10,   // inference threads
  "model_unload_grace_seconds": 600,  // ceiling on the model-swap wait
  "unload_poll_seconds": 10,          // how often to check whether it is gone
  "keep_loaded_seconds": "10m"        // hold the model between batches of a stage
}
```

- `num_ctx`, `num_gpu`, and `num_thread` only apply to Ollama.
- `keep_loaded_seconds` keeps a model in RAM **between batches of the same
  stage**, then releases it on that stage's last batch. Measured on this box, a
  cold load is 393 s of a 923 s batch — 43% of it — because Ollama mmaps lazily
  and 19 GB faults in from disk during the first prefill. Without this, every
  batch would pay that again.
- `model_unload_grace_seconds` is now a **ceiling, not a fixed cost**. Every
  call passes `keep_alive=0`, so on a model swap the script polls `ollama ps`
  every `unload_poll_seconds` and continues as soon as the previous model has
  actually left RAM — usually seconds. It only waits the full grace if
  `ollama ps` cannot be reached. Set it to `0` for cloud providers, where there
  is nothing to unload.

### Email — `email` section

```jsonc
"email": {
  "sender_email": "you@gmail.com",
  "receiver_email": "you@vt.edu",   // or a list: ["a@x.edu", "b@y.edu"]
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 465,
  "send_hour_start": 8,
  "send_hour_end": 24
}
```

The digest is only emailed inside this window; runs finishing earlier hold until
`send_hour_start`. Set `PASSWORD` (a Gmail app password) in `.env` — the script
exits at startup if it is missing, rather than discovering it hours later.

`receiver_email` takes a single address or a list. The digest is sent as a
`multipart/alternative` message (plain text plus HTML) with an explicit UTF-8
charset, so accented characters and en-dashes in VT event titles arrive intact.

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

### Output — `output` section

```jsonc
"output": {
  "mode": "html+email"   // "html" | "email" | "html+email"
}
```

Which outputs a run produces:

| mode | HTML report | Email | Notes |
| --- | --- | --- | --- |
| `html` | written | — | No send-window wait; the run ends as soon as the report is on disk. |
| `email` | — | sent | The digest carries no "full list" link, because no report exists to link to. |
| `html+email` | written | sent | Default. The report is written *before* the send-window hold, so it is servable immediately. |

`both` is accepted as an alias for `html+email`. An unrecognized value exits
with an explanation rather than guessing. `--dry-run` suppresses the email in
any mode.

## Interests file

`interests.json` (path set via `files.interests`) lists the "high interest"
criteria that determine what lands in the email:

```json
{
  "criteria": [
    { "text": "Offers free food or free items",
      "auto_match_flags": ["FREE FOOD", "FREE ITEMS / GIVEAWAYS"] },
    "Career / job fairs or recruiting events"
  ]
}
```

A criterion is either a plain string or an object. Both forms can be mixed, and
the criteria are numbered in order — that numbering is what the model returns,
so a reply is a couple of digits per event instead of a restatement.

- **`text`** — the criterion, rendered verbatim into every reviewer's prompt.
- **`auto_match_flags`** *(optional)* — facets the source states outright. An
  event carrying one of these flags matches **without a model call**:
  events.vt.edu publishes free food and giveaways as CSS classes on its listing,
  so that is a fact to read, not a judgement to make. Currently emitted flags
  are `FREE FOOD`, `FREE ITEMS / GIVEAWAYS`, `free parking` and `free admission`.

Edit this one list to change what counts as high-interest.

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
`events_vt`, `career_vt`, `career_fairs`, `gobblerconnect`, or `news`. Unknown
types are skipped.

The `type` must match the page, not just the domain. `career_vt` parses the
uConnect events archive at `career.vt.edu/events/` (`li.event_item`), while
`career_fairs` parses the semester schedule tables at
`career.vt.edu/resources/career-fairs/` — the same site, but markup with no
`li.event_item` in it. Giving a page the wrong parser is a silent failure: the
source collects nothing and prints only `WARNING: <id> returned no events this
week`. In `web_search` mode only the `name` and `url` are used (the model
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

Set `GEMINI_API_KEY` in `.env`. Uses the current `google-genai` SDK
(`pip install google-genai`) with `response_mime_type: application/json`.

### OpenAI

```jsonc
"worker": { "provider": "openai", "url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY", "model": "gpt-4o" }
```

Set `OPENAI_API_KEY` in `.env`. Classification requests use
`response_format: {"type": "json_object"}`.

### Claude (Anthropic)

```jsonc
"worker": { "provider": "claude", "url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY", "model": "claude-3-5-sonnet" }
```

Set `ANTHROPIC_API_KEY` in `.env`. `provider` may be `"claude"` or
`"anthropic"` — both are accepted. JSON replies use assistant prefill.

Any of the four providers can be used for the worker *and/or* the judges, and
they may be mixed (e.g. an Ollama worker with a Gemini judge).

### Local vs. cloud

Both paths are first class, tuned for opposite constraints:

| | Ollama | Cloud |
| --- | --- | --- |
| Model swaps | `ollama ps` polled until the old model leaves RAM | nothing to unload — set the grace to `0` |
| JSON | Ollama's `format: json` constrained sampling | native JSON mode per provider |
| Reasoning | `model_thinking` per model | provider default |
| Failures | retried with backoff | retried with backoff, plus a per-request timeout |

A run's cost is dominated by generated tokens, and classification generates a
few hundred per batch regardless of backend — so the same config that takes
under an hour locally takes a couple of minutes on a cloud provider.

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
