# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Single-script pipeline that scrapes Virginia Tech event/notice pages plus GobblerConnect, sends the scraped text through a two-stage LLM pipeline (worker extraction + judge review) to identify and filter events, writes a local HTML report, and emails high-interest matches via Gmail SMTP.

## Commands

- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run: `python vt_events_reminder.py`
- No tests, linting, or CI are configured in this repo.

## Requirements to run

- No browser required. In `builtin` mode every source is fetched with `requests` — see `get_vt_data()` below.
- The default LLM backend is Ollama (`models.worker.judge1.judge2` in `config.json`), which requires `ollama serve` running locally with all three models pulled: worker `gemma4:31b`, judge1 `muse-glimmer:30b` and judge2 `qwen3.8:27b`. They are deliberately from three different labs (Google / Meta / Alibaba) so no two stages share a training lineage, and therefore no two share a blind spot. All three are thinking-capable. Reasoning stays on for the worker; it is turned **off** for both judges via `JUDGE_THINKING` (see below). Each is ~17–19 GB and loads alone, so only one need fit in RAM at a time. Each stage can be a different provider — see `call_model()`.
- Inference is pinned to CPU: every Ollama call passes `num_gpu` (`NUM_GPU = 0`) alongside `num_ctx` and `num_thread`. The GPU is a 2 GB GTX 1050 — far too small for these ~18 GB models — but left to itself Ollama's scheduler auto-offloads a few layers plus their KV-cache slice, which buys no speed (tokens still wait on the CPU layers) and can OOM the card when the desktop is using it. With `num_gpu=0`, `ollama ps` should report 100% CPU. Model choice here is made on quality, not speed — the script is run overnight.
- `NUM_THREAD` (10) overrides llama.cpp's default of one thread per physical core (6 on this i7-8700). Measured decode on `gemma4:31b` (150 tokens/arm): 6 → 1.512 tok/s, 10 → 1.524 / 1.526, 12 → 1.378 / 1.380. **12 is reproducibly ~10% slower than 10** despite splitting evenly across cores — llama.cpp syncs all threads at each layer boundary, so with no logical CPU spare for the kernel, the ollama server and interrupts, one descheduled thread stalls the whole graph. Leaving 2 free costs nothing: hyperthreads add no memory bandwidth (30.2 GB/s at 12 vs 31.4 GB/s at 6). Re-measure before changing.
- `JUDGE_THINKING` maps each judge model to whether reasoning is on. Both are `False`. Judges only match titles against the scraped list, the task reasoning helps least, and measured runs had each judge generating ~10k tokens to emit a few hundred tokens of title lists. Flip either entry back to `True` to restore reasoning for that judge alone. The worker is never in this map — it writes the event list and keeps reasoning.
- `NUM_CTX` (32768) is used for every Ollama call. Judge prompts carry the scraped content *plus* the worker's full JSON output; measured against real runs the high-water mark is ~19k tokens, so 32768 leaves headroom. 65536 was reserving 7.5 GB of KV cache (5120 MiB non-SWA + 2400 MiB SWA) for a context less than a third that size.
- To use a cloud provider (Gemini/OpenAI/Claude), set each stage's `provider`/`url`/`model` in `config.json`, create the matching `api_key_env` key in `.env`, and typically set `scraping.mode` to `web_search` so the model browses the sources itself.

## Where the run time actually goes

A full run is dominated by **token generation**, not scraping or prefill. Measured from `~/.ollama/ollama-server.log` (`slot print_timing` lines record every call):

- Decode runs at **0.67–1.26 tok/s** for all three models — none is an outlier. The roofline is memory bandwidth: ~31 GB/s measured, and a 19 GB Q4_K_M model needs a full pass per token, so ~1.65 tok/s is the ceiling. The i7-8700 has AVX2 but no AVX-512/VNNI, so Q4_K dequant is compute-bound on top of that.
- Calls have generated **7,000–15,900 tokens** each. The useful JSON is ~6,000 tokens for the worker and a few hundred for each judge; the rest is reasoning. Thinking is on by default for any thinking-capable model unless `think=False` is passed, and it is what makes a call take hours. `_log_ollama_cost()` prints prompt/generated token counts per stage — check those numbers first when a run is slow.
- Individual `/api/generate` calls have run 2h17m to 8h20m. A four-call run is ~16 h, plus 30 min of `MODEL_UNLOAD_GRACE_SECONDS` sleeps.
- Prefill varies 5.4–40 tok/s between runs because Ollama launches llama-server with `--load-mode none` (lazy mmap): the 19 GB of weights fault in from the SATA SSD *during* the first prefill batch. Three ~18 GB models cycling through ~27 GB of page cache means every load is cold.

When a run seems slow, check the generated token count in the log before suspecting a particular model.

## Architecture

Everything lives in `vt_events_reminder.py`, run top-to-bottom from `__main__`:

1. **`get_vt_data()`** — the two scrape modes from `SCRAPE_MODE`:
   - `builtin` — collects this week's (Mon–Sun, `week_bounds()`) events from each source in `sources.json`, each via a dedicated parser (selected by `_builtin_collector` from the source `type`) that emits labelled `TITLE:/WHEN:/LOCATION:/...` blocks rather than raw page text. Returns `(combined_content, sources)` where `sources` maps URL → collected text, used later to attribute each event to a source link.
   - `fetch_events_vt()` — month listing pages (`events.vt.edu/events/{YYYY}/{MM}.html`), parsing `li.event-page`. **The listing `<li>` CSS classes are authoritative facets** — `features_-free-food`, `features_-giveaways`, `admission_-free`, `categories_-*`, `locations_-*` — so free-food/free-item detection comes from the site, not from LLM inference. Each in-week event's detail page is then fetched for its description/time/location, and run through `_strip_boilerplate()` before the `DETAILS:` character budget is applied — that removes share widgets, "View On Map", the duplicated "Accessibility Contact:" line, and the trailing tag cloud, cutting ~25% of the detail text. The tag cloud is safe to drop because free-food/giveaway signal is read from the listing CSS classes into `FLAGS:`, never from that text; verify that still holds if the tag list is ever used for anything.
   - `fetch_career_vt()` — `career.vt.edu/events/`, parsing `li.event_item` (uConnect, server-rendered).
   - `fetch_gobblerconnect()` — the **CampusGroups JSON web service** at `/mobile_ws/v17/mobile_events_list?range=&limit=`, which is the same endpoint the JS front end calls. No browser needed; returns `eventName`/`eventDates`/`eventLocation`/`eventCategory`/`clubName` directly.
   - `fetch_news_vt()` — notice/article headlines from `news.vt.edu/notices.html` and `news.vt.edu`.

   Date filtering uses `_parse_dates()` + `_in_week()`; multi-date text is treated as a span, and spans longer than `EVERGREEN_MAX_DAYS` (60) are dropped so always-on listings don't appear in every digest. A source that yields nothing emits `(no entries found for this week)` — it never falls back to dumping page text, which previously flooded the prompt with navigation boilerplate.
   - `web_search` — no local fetch; instead the worker model is handed the list of source URLs (from `sources.json`) and uses its own web-search/browsing tools to look them up. `sources` is returned empty and the deterministic completeness check is skipped.
2. **Worker phase** (`worker_call` → `call_model`) — sends the scraped content + prompt (`get_prompt`) to the worker LLM, asking for strict JSON: `{"all_events": [...], "filtered_events": [...]}`. `extract_json()` strips markdown fences/conversational text before `json.loads`. On failure the script retries indefinitely with a 5s backoff (`while events is None`).
3. **Judge phase** (`run_judge`, any provider) — `JUDGE1_SPEC` and `JUDGE2_SPEC` (from `models.judge1`/`models.judge2`; `stages.num_judges` is 0, 1, or 2) each independently review the worker's output against the scraped content. **Judges report, they never rewrite**: each returns `{"missing_events", "wrongly_included", "should_be_filtered", "notes"}` with no `corrected_output`. With two reviewers a wholesale rewrite from each would be incoherent (whose wins?), so the worker stays the single writer. A judge that errors is skipped and the run continues with whichever reports arrived. Both default judges run with reasoning off (`JUDGE_THINKING`); if judge quality regresses, that is the first thing to put back.
4. **Merging verdicts** (`merge_judgements`) — the two error types are weighted differently because their costs are asymmetric. Additions (`missing_events`, `should_be_filtered`) take the **union** — one judge spotting a missed free-food event is enough. Removals (`wrongly_included`) require **unanimous** agreement, so a single overzealous reviewer cannot prune a real hit; disagreements are reported as `contested_removals` and kept. `build_critique()` renders the merged verdict, and the worker is re-queried **once** with it (`worker_call(get_revision_prompt(...))`). **The revision returns additions only.** It is given the titles it already has (a few tokens each) via `get_revision_prompt()` and returns `{"new_events", "new_filtered"}`; `apply_revision()` folds them in, de-duplicating on `_normalize_text` and back-filling any high-interest event that was missing from `all_events`. Unanimous removals are applied in code by title match — never by asking the model to reproduce the list without them. This exists because re-emitting the whole list was measured at 12,629 generated tokens and 4h17m, over half of an 8h run. If the revision comes back empty, the original output is kept.
5. **Deterministic completeness check** (`count_source_events`) — counts `^TITLE: ` blocks in the scraped content and compares against `len(all_events)`. Counting is something code does perfectly and LLMs do badly, so this never goes to a model; the shortfall is fed into the critique.
6. **Model swap guard** (`wait_for_model_unload`) — switching models sleeps `MODEL_UNLOAD_GRACE_SECONDS` (10 min) whenever the requested model differs from the last one used, guaranteeing the previous model has fully unloaded before the next load begins. The pipeline makes 3 model loads per run (4 if the worker revises), so this dominates wall-clock time — by design, since the script runs overnight.
7. **`create_local_html()`** — renders `all_events` to `HTML_FILE_PATH`, matching each event's title against normalized source text (`_normalize_text`) to link back to its origin URL, and stripping emojis (`strip_emojis`) from displayed fields.
8. **`send_filtered_email()`** — emails `filtered_events` via Gmail SMTP_SSL using `SENDER_EMAIL`/`PASSWORD` from config + `.env`. Gated by **`wait_for_send_window()`**: cron starts the run at midnight and it can finish before dawn, so the email is held until `SEND_HOUR_START` (08:00) and sent immediately if the run already finished inside the 08:00–24:00 window. The HTML report is written *before* the hold, so the full list is on disk and servable as soon as the pipeline ends, even while the email waits.

Filtering criteria are defined **once** in `interests.json` (`criteria`, path set via `config.json` → `files.interests`) and rendered into both the worker prompt and every judge prompt via `_criteria_block()`. Edit that one list to change them — the prompts can no longer drift out of sync. Sources live in `sources.json` (`files.sources`).

## Configuration — where everything lives

Non-secret settings are in `config.json` (which the script reads through the module-level `CONFIG_PATH` variable, overridable via the `CONFIG_PATH` env var or `--config` on the command line). Secrets are in a `.env` file next to it (overridable via `DOTENV_PATH`), loaded with `python-dotenv`. Interests come from `interests.json`, sources from `sources.json`. The script's `load_config()`/`apply_config()` expose everything as module globals so the rest of the file reads them unchanged:

- `models.worker` / `models.judge1` / `models.judge2` → `MODEL_SPECS`, `WORKER_SPEC`, `JUDGE1_SPEC`, `JUDGE2_SPEC` (each a `{provider, url, api_key_env, model}` dict). With `stages.use_same_model_for_all = true`, judges reuse the worker spec.
- `stages.num_judges` → `NUM_JUDGES` (0, 1, or 2); `stages.judge_thinking` → `JUDGE_THINKING` (Ollama reasoning only)
- `scraping.mode` → `SCRAPE_MODE` (`builtin` | `web_search`); `files.*` resolves `interests.json`/`sources.json` → `INTEREST_CRITERIA`, `SOURCES`
- `ollama.*` → `NUM_CTX`, `NUM_GPU`, `NUM_THREAD`, `MODEL_UNLOAD_GRACE_SECONDS`
- `email.*` → `SENDER_EMAIL`, `RECEIVER_EMAIL`, `SMTP_SERVER`, `SMTP_PORT`, `SEND_HOUR_START`, `SEND_HOUR_END`
- `report.*` → `HTML_FILE_PATH`, `REPORT_URL` (built from `server_ip`/`server_port`/filename)
- `scraping.*` tuning → `HTTP_TIMEOUT`, `POLITE_DELAY`, `EVERGREEN_MAX_DAYS`, `GOBBLERCONNECT_PAGE_SIZE`, `GOBBLERCONNECT_MAX_EVENTS`, `EVENT_DETAIL_LIMIT`

## Credentials (in `.env` — handle with care)

`PASSWORD` (Gmail app password) and `GEMINI_API_KEY` live only in `.env`, which is git-ignored. `SENDER_EMAIL`/`RECEIVER_EMAIL` are non-secret and live in `config.json`'s `email` section. Never commit real credential values; `.env` is covered by `.gitignore`. The previously-committed app password should be treated as compromised and rotated.

## Run logging

Every phase boundary prints through `_stamp()` — `[HH:MM:SS | +N min] TAG message`, flushed immediately because cron redirects stdout to a file and Python otherwise block-buffers it for hours. Tags: `SCRAPE` (start/done), `CALL` (per model call), `LOAD`/`DONE` around each Ollama call, `UNLOAD` around each `MODEL_UNLOAD_GRACE_SECONDS` sleep, `REPORT`, `EMAIL` (including whether it held for the send window or sent immediately), `FINISH`. `_log_ollama_cost()` adds prompt/generated token counts per stage. Read the log top to bottom to see exactly where a slow run went.

## Known fragility

- No error handling for malformed LLM JSON beyond the retry loop in `__main__` — a persistently broken worker response will retry forever.
- Cloud providers need their SDK (`openai`, `anthropic`, `google-generativeai`) installed and a populated `api_key_env` in `.env`; Ollama ignores its key. A provider without a key raises at call time.
- The scrapers depend on site-specific CSS classes (`li.event-page`, `li.event_item`) and on the undocumented GobblerConnect `mobile_ws` endpoint. A VT redesign breaks these silently — a source will simply report 0 entries rather than error, so watch the per-source counts printed by `get_vt_data()`.
- `HTML_FILE_PATH` is an absolute path (set in `config.json` → `report.html_file_path`) — update it if the repo moves.
