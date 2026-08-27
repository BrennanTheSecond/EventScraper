import html
import os
import re
import sys
import requests
from bs4 import BeautifulSoup
import smtplib
import ssl
import json
from datetime import date, datetime, timedelta

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# To use Ollama, you need the 'ollama' python library: pip install ollama
try:
    import ollama
except ImportError:
    ollama = None

import time

# ==========================================
# CONFIGURATION
#
# Non-secret settings live in config.json next to this script (see
# CONFIG_PATH): model definitions, scrape mode, email + report settings,
# scraping tunables, and Ollama tuning. Secrets -- Gmail app password, model
# API keys -- live in a .env file next to it. Point CONFIG_PATH at a different
# file (via env var or --config on the command line) to override, e.g. for a
# test environment.
#
# Every model stage (scraper, worker, judge1, judge2) is a dict with:
#   provider : "ollama" | "gemini" | "claude" | "openai"
#   url      : the endpoint the provider is reached at (you fill this in)
#   api_key_env : name of the env var (in .env) holding this model's key
#   model    : the model name to send
#
# Ollama ignores api_key_env entirely -- only url + model matter.
# ==========================================
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config.json")


def _json_path(base_name, config_dir):
    """Resolve a relative filename against the config file's directory."""
    if os.path.isabs(base_name):
        return base_name
    return os.path.join(config_dir, base_name)


def load_config():
    """Read non-secret settings from config.json and secrets from .env.

    Returns a dict so callers can inspect; the values are also exposed as
    module-level globals by the assignment block that follows.
    """
    cfg_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    # --config on the command line wins over the environment variable.
    if "--config" in sys.argv:
        flag = sys.argv.index("--config")
        if flag + 1 < len(sys.argv):
            cfg_path = sys.argv[flag + 1]
    if not os.path.exists(cfg_path):
        sys.exit(f"Config file not found: {cfg_path}\n"
                 f"Set CONFIG_PATH (or copy config.json next to the script).")

    with open(cfg_path, "r") as fh:
        cfg = json.load(fh)

    # Secrets from .env next to the config (or wherever DOTENV_PATH points).
    env_path = os.environ.get("DOTENV_PATH",
                              os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                                           ".env"))
    if load_dotenv and os.path.exists(env_path):
        load_dotenv(env_path, override=False)

    # Load the separate interests and sources JSON files (relative to config).
    config_dir = os.path.dirname(os.path.abspath(cfg_path))
    files = cfg.get("files", {})

    interests_path = _json_path(files.get("interests", "interests.json"), config_dir)
    sources_path = _json_path(files.get("sources", "sources.json"), config_dir)

    with open(interests_path, "r") as fh:
        cfg["_interests"] = json.load(fh)["criteria"]
    with open(sources_path, "r") as fh:
        cfg["_sources"] = json.load(fh)["sources"]

    return cfg


def apply_config(cfg):
    """Expose the loaded config as module-level globals."""
    models = cfg["models"]
    stages = cfg.get("stages", {})
    email = cfg["email"]
    report = cfg["report"]
    scrape = cfg["scraping"]
    ollama_cfg = cfg.get("ollama", {})

    global MODEL_SPECS, USE_SAME_MODEL, NUM_JUDGES, JUDGE_THINKING
    global SCRAPE_MODE, SOURCES
    global WORKER_SPEC, JUDGE1_SPEC, JUDGE2_SPEC
    global NUM_CTX, NUM_GPU, NUM_THREAD
    global SENDER_EMAIL, RECEIVER_EMAIL, PASSWORD
    global HTML_FILE_PATH, REPORT_URL
    global SEND_HOUR_START, SEND_HOUR_END
    global SMTP_SERVER, SMTP_PORT
    global INTEREST_CRITERIA
    global MODEL_UNLOAD_GRACE_SECONDS
    global HTTP_TIMEOUT, POLITE_DELAY, EVERGREEN_MAX_DAYS
    global GOBBLERCONNECT_PAGE_SIZE, GOBBLERCONNECT_MAX_EVENTS, EVENT_DETAIL_LIMIT

    MODEL_SPECS = models
    WORKER_SPEC = models["worker"]
    USE_SAME_MODEL = stages.get("use_same_model_for_all", False)
    if USE_SAME_MODEL:
        JUDGE1_SPEC = JUDGE2_SPEC = WORKER_SPEC
    else:
        JUDGE1_SPEC = models.get("judge1", WORKER_SPEC)
        JUDGE2_SPEC = models.get("judge2", WORKER_SPEC)
    NUM_JUDGES = max(0, min(int(stages.get("num_judges", 2)), 2))
    JUDGE_THINKING = stages.get("judge_thinking", {}) or {}

    SCRAPE_MODE = scrape.get("mode", "builtin")
    SOURCES = cfg.get("_sources", [])

    NUM_CTX = ollama_cfg.get("num_ctx", 32768)
    NUM_GPU = ollama_cfg.get("num_gpu", 0)
    NUM_THREAD = ollama_cfg.get("num_thread", 10)
    # 0 is ideal for paid/proxied model providers (no local process to unload);
    # ~10 minutes is right for Ollama spinning large local models on/off disk.
    MODEL_UNLOAD_GRACE_SECONDS = ollama_cfg.get("model_unload_grace_seconds", 600)

    SENDER_EMAIL = email["sender_email"]
    RECEIVER_EMAIL = email["receiver_email"]
    PASSWORD = os.environ.get("PASSWORD")
    if not PASSWORD:
        print("WARNING: PASSWORD not set in .env -- email will fail.")

    HTML_FILE_PATH = report["html_file_path"]
    server_ip = report.get("server_ip", "")
    server_port = report.get("server_port", "6060")
    base_name = os.path.basename(HTML_FILE_PATH)
    REPORT_URL = (f"http://{server_ip}:{server_port}/{base_name}"
                  if server_ip else "")

    SEND_HOUR_START = email.get("send_hour_start", 8)
    SEND_HOUR_END = email.get("send_hour_end", 24)
    SMTP_SERVER = email.get("smtp_server", "smtp.gmail.com")
    SMTP_PORT = email.get("smtp_port", 465)

    INTEREST_CRITERIA = cfg.get("_interests", [])

    HTTP_TIMEOUT = scrape.get("http_timeout", 20)
    POLITE_DELAY = scrape.get("polite_delay", 0.5)
    EVERGREEN_MAX_DAYS = scrape.get("evergreen_max_days", 60)
    GOBBLERCONNECT_PAGE_SIZE = scrape.get("gobblerconnect_page_size", 100)
    GOBBLERCONNECT_MAX_EVENTS = scrape.get("gobblerconnect_max_events", 600)
    EVENT_DETAIL_LIMIT = scrape.get("event_detail_limit", 1800)


_CFG = load_config()
apply_config(_CFG)

# Email context (ssl) -- built once, reused for every send.
CONTEXT = ssl.create_default_context()


def _criteria_block(indent="   "):
    return "\n".join(f"{indent}- {c}" for c in INTEREST_CRITERIA)


def _spec_api_key(spec):
    """Secret key for a model spec, read from .env via its api_key_env name."""
    if not spec:
        return None
    name = spec.get("api_key_env")
    if not name:
        return None
    return os.environ.get(name)


def _is_local(spec):
    """True if a model spec is served locally (Ollama) vs. a cloud provider."""
    return bool(spec) and spec.get("provider", "").lower() in ("ollama", "local")


# Ollama model swap guard.
# This machine only has enough RAM for the 70B worker model. Starting to load a
# different model before the previous one has fully unloaded can crash the load,
# so before any model switch we sleep to guarantee the old model is evicted.
_last_ollama_model = None

# Wall-clock origin for the elapsed column in every progress line. Set at import
# so the numbers line up with cron's start time rather than with __main__.
_RUN_STARTED = time.time()


def _stamp(message):
    """One timestamped progress line, flushed immediately.

    Every run goes to a log file via cron redirect, and Python block-buffers
    stdout when it is not a terminal -- without flush the whole run's output
    would appear only at exit, which is useless for watching a job that takes
    hours. The clock time answers "when did this happen"; the elapsed column
    answers "how long into the run", which is what the timings are compared on.
    """
    clock = datetime.now().strftime("%H:%M:%S")
    elapsed = (time.time() - _RUN_STARTED) / 60
    print(f"[{clock} | +{elapsed:7.1f} min] {message}", flush=True)


def wait_for_model_unload(model_name):
    """Sleep before loading `model_name` if a different Ollama model was the last
    one used, guaranteeing the previous model has fully unloaded from RAM."""
    global _last_ollama_model
    if _last_ollama_model and _last_ollama_model != model_name:
        _stamp(f"UNLOAD  waiting {MODEL_UNLOAD_GRACE_SECONDS // 60} min for "
               f"'{_last_ollama_model}' to leave RAM before loading '{model_name}'")
        started = time.time()
        time.sleep(MODEL_UNLOAD_GRACE_SECONDS)
        _stamp(f"UNLOAD  done, '{_last_ollama_model}' unloaded "
               f"({(time.time() - started) / 60:.1f} min)")
    _last_ollama_model = model_name


def wait_for_send_window():
    """Hold the email until the clock is inside the send window.

    Returns a short description of what happened, for the caller to log.
    """
    now = datetime.now()
    if SEND_HOUR_START <= now.hour < SEND_HOUR_END:
        return (f"finished at {now:%H:%M}, already inside the "
                f"{SEND_HOUR_START:02d}:00-{SEND_HOUR_END:02d}:00 window -- sending immediately")

    target = now.replace(hour=SEND_HOUR_START, minute=0, second=0, microsecond=0)
    if now.hour >= SEND_HOUR_END:
        # Only reachable if SEND_HOUR_END is set below midnight; the next
        # opening is then tomorrow morning.
        target += timedelta(days=1)
    wait_seconds = max(0.0, (target - now).total_seconds())
    _stamp(f"EMAIL   finished at {now:%H:%M}, before the send window opens at "
           f"{SEND_HOUR_START:02d}:00 -- holding {wait_seconds / 60:.1f} min")
    time.sleep(wait_seconds)
    return (f"waited {wait_seconds / 60:.1f} min for the "
            f"{SEND_HOUR_START:02d}:00 send window")

# ---------------------------------------------------------------------------
# Scraping helpers
#
# Every VT source below is reachable without a browser: events.vt.edu,
# news.vt.edu and career.vt.edu are server-rendered, and GobblerConnect -- the
# only genuinely JS-rendered page -- exposes the same data as JSON through the
# CampusGroups mobile web service its own front end calls. Nothing here needs
# Selenium or a headless Chromium.
# ---------------------------------------------------------------------------
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; vt-events-reminder/1.0)"}
HTTP_TIMEOUT = 20
POLITE_DELAY = 0.5          # seconds between event-detail page fetches
EVERGREEN_MAX_DAYS = 60     # drop "events" that span longer than this (always-on listings)

_MONTH_RE = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_RE = re.compile(r"\b(" + _MONTH_RE + r")\.?\s+(\d{1,2})(?:,\s*(\d{4}))?\b", re.I)

# events.vt.edu encodes facets as CSS classes on each listing <li>. These are
# authoritative -- far better than asking the LLM to infer "is there free food".
EVENTS_VT_FLAGS = {
    "features_-free-food": "FREE FOOD",
    "features_-giveaways": "FREE ITEMS / GIVEAWAYS",
    "features_-free-parking": "free parking",
    "admission_-free": "free admission",
}


def _get(url, as_json=False):
    resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
    resp.raise_for_status()
    return resp.json() if as_json else resp.text


def _clean(text):
    # Some VT pages double-encode entities ("&amp;amp;"), so unescape twice.
    return " ".join(html.unescape(html.unescape(text or "")).split())


# Page chrome that every events.vt.edu detail page carries. None of it describes
# the event, and on a normal week the DETAILS: fields are ~a third of the whole
# prompt -- which on this CPU is paid for twice, once in prefill and again as
# deeper attention on every generated token.
_BOILERPLATE_RE = [
    re.compile(r"Share on Facebook Share on X Copy address link to clipboard"),
    re.compile(r"View On Map"),
    re.compile(r"Submit a new event \(PID Required\) CREATE EVENT"),
    # Verbatim repeat of the Contact line directly above it. Only cut when the
    # "About the Event" anchor is there to bound it, so a page with an unusual
    # layout keeps its text rather than losing the description.
    re.compile(r"Accessibility Contact:.*?(?=About the Event)"),
]

# The trailing tag cloud duplicates the listing facets already emitted as
# CATEGORY/AUDIENCE/DEPARTMENT/FLAGS. Dropping it costs no free-food signal:
# that is read from the listing <li> CSS classes into FLAGS, never from here.
_TAG_CLOUD_MAX = 400


def _strip_boilerplate(text):
    """Remove events.vt.edu page furniture from a detail-page body."""
    for pattern in _BOILERPLATE_RE:
        text = pattern.sub(" ", text)
    # Cut from the LAST "Tags " and only when what follows is short enough to be
    # a real tag cloud, so the word "tags" inside a description is left alone.
    idx = text.rfind(" Tags ")
    if idx != -1 and len(text) - idx <= _TAG_CLOUD_MAX:
        text = text[:idx]
    return " ".join(text.split())


def week_bounds(today=None):
    """Monday-Sunday range containing `today`."""
    today = (today or datetime.now()).date()
    start = today - timedelta(days=today.weekday())
    return start, start + timedelta(days=6)


def _parse_dates(text, default_year):
    """All calendar dates mentioned in `text`, in order of appearance."""
    found = []
    for m in _DATE_RE.finditer(text or ""):
        month_token = m.group(1)[:3].title()
        try:
            month = datetime.strptime(month_token, "%b").month
            year = int(m.group(3)) if m.group(3) else default_year
            found.append(date(year, month, int(m.group(2))))
        except ValueError:
            continue
    return found


def _in_week(dates, week_start, week_end):
    """True if a single date lands in the week, or a date range overlaps it.

    Multi-date text is treated as a span (start .. end). Spans longer than
    EVERGREEN_MAX_DAYS are always-on listings rather than real weekly events,
    so they are excluded to keep them out of every single digest.
    """
    if not dates:
        return False
    first, last = min(dates), max(dates)
    if (last - first).days > EVERGREEN_MAX_DAYS:
        return False
    return first <= week_end and last >= week_start


def _facet_values(classes, prefix):
    return [c.split("_-", 1)[1].replace("-", " ")
            for c in classes if c.startswith(prefix)]


def _block(**fields):
    """Render one event as a compact labelled block for the LLM prompt."""
    lines = [f"{k}: {v}" for k, v in fields.items() if v]
    return "\n".join(lines)


def _event_detail_text(url, limit=EVENT_DETAIL_LIMIT):
    """Body text of an events.vt.edu detail page (description, time, location)."""
    try:
        soup = BeautifulSoup(_get(url), "html.parser")
    except Exception:
        return ""
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    main = soup.select_one("main") or soup.select_one("[role=main]") or soup
    # Strip before truncating, so the character budget goes to description text
    # rather than to share widgets and the tag cloud.
    return _strip_boilerplate(_clean(main.get_text(" ", strip=True)))[:limit]


def fetch_events_vt(week_start, week_end):
    """events.vt.edu -- month listing pages plus per-event detail pages."""
    blocks, seen = [], set()
    months = []
    for d in (week_start, week_end):
        if (d.year, d.month) not in months:
            months.append((d.year, d.month))

    for year, month in months:
        url = f"https://events.vt.edu/events/{year}/{month:02d}.html"
        try:
            soup = BeautifulSoup(_get(url), "html.parser")
        except Exception as e:
            blocks.append(f"[Error fetching {url}: {e}]")
            continue

        for li in soup.select("li.event-page"):
            text = _clean(li.get_text(" ", strip=True))
            if not _in_week(_parse_dates(text, year), week_start, week_end):
                continue
            anchor = li.select_one("a[href]")
            link = anchor["href"] if anchor else url
            if link in seen:
                continue
            seen.add(link)

            # "Event Item <title> , event Date: Aug 14, 2026"
            title = text.split("Event Item", 1)[-1].split(", event", 1)[0].strip(" ,")
            when = text.split("Date:", 1)[1].strip() if "Date:" in text else ""
            classes = li.get("class") or []

            time.sleep(POLITE_DELAY)
            blocks.append(_block(
                TITLE=title,
                WHEN=when,
                CATEGORY=", ".join(_facet_values(classes, "categories_-")),
                LOCATION=", ".join(_facet_values(classes, "locations_-")),
                DEPARTMENT=", ".join(_facet_values(classes, "departments_-")),
                AUDIENCE=", ".join(_facet_values(classes, "audiences_-")),
                FLAGS=", ".join(label for cls, label in EVENTS_VT_FLAGS.items()
                                if cls in classes),
                LINK=link,
                DETAILS=_event_detail_text(link),
            ))
    return blocks


def fetch_career_vt(week_start, week_end):
    """career.vt.edu -- uConnect events archive (server-rendered)."""
    url = "https://career.vt.edu/events/"
    try:
        soup = BeautifulSoup(_get(url), "html.parser")
    except Exception as e:
        return [f"[Error fetching {url}: {e}]"]

    blocks, seen = [], set()
    for li in soup.select("li.event_item"):
        text = _clean(li.get_text(" ", strip=True))
        if not _in_week(_parse_dates(text, week_start.year), week_start, week_end):
            continue
        anchor = li.select_one("a[href]")
        link = anchor["href"] if anchor else url
        if link in seen:
            continue
        seen.add(link)
        title = text.split("Event:", 1)[-1].strip() if "Event:" in text else text
        blocks.append(_block(
            TITLE=_clean(title)[:160],
            CATEGORY="Career",
            LINK=link,
            DETAILS=text[:700],
        ))
    return blocks


def fetch_gobblerconnect(week_start, week_end,
                         page_size=GOBBLERCONNECT_PAGE_SIZE,
                         max_events=GOBBLERCONNECT_MAX_EVENTS):
    """GobblerConnect -- CampusGroups JSON web service (no browser needed).

    The /events page renders client-side from this same endpoint, so we read the
    structured records directly instead of driving a headless browser.
    """
    base = "https://gobblerconnect.vt.edu/mobile_ws/v17/mobile_events_list"
    blocks, seen = [], set()

    for offset in range(0, max_events, page_size):
        url = (f"{base}?range={offset}&limit={page_size}"
               f"&filter4_contains=OR&order=undefined")
        try:
            rows = _get(url, as_json=True)
        except Exception as e:
            blocks.append(f"[Error fetching GobblerConnect: {e}]")
            break
        if not rows:
            break

        for row in rows:
            fields = (row.get("fields") or "").split(",")
            rec = dict(zip(fields, [row.get(f"p{i}") for i in range(len(fields))]))
            if rec.get("displayType") != "event" or not rec.get("eventName"):
                continue
            event_id = rec.get("eventId")
            if event_id in seen:
                continue

            when = _clean(BeautifulSoup(rec.get("eventDates") or "", "html.parser")
                          .get_text(" ").replace("–", "-"))
            if not _in_week(_parse_dates(when, week_start.year), week_start, week_end):
                continue
            seen.add(event_id)

            event_url = rec.get("eventUrl") or ""
            if event_url.startswith("/"):
                event_url = "https://gobblerconnect.vt.edu" + event_url
            blocks.append(_block(
                TITLE=_clean(rec.get("eventName")),
                WHEN=when,
                LOCATION=_clean(rec.get("eventLocation")),
                CATEGORY=_clean(rec.get("eventCategory")),
                HOST=_clean(rec.get("clubName")),
                LINK=event_url,
            ))
    return blocks


def fetch_news_vt(url):
    """news.vt.edu notices / stories -- link text is the headline."""
    try:
        soup = BeautifulSoup(_get(url), "html.parser")
    except Exception as e:
        return [f"[Error fetching {url}: {e}]"]

    blocks, seen = [], set()
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        if not re.search(r"/notices/|/articles/20", href):
            continue
        title = _clean(anchor.get_text(" ", strip=True))
        if len(title) < 15 or title in seen:
            continue
        seen.add(title)
        if href.startswith("/"):
            href = "https://news.vt.edu" + href
        blocks.append(_block(TITLE=title, LINK=href))
    return blocks


def _builtin_collector(source, week_start, week_end):
    """Return the right fetch callable for one sources.json entry."""
    url = source["url"]
    stype = source.get("type")
    if stype == "events_vt":
        return lambda: fetch_events_vt(week_start, week_end)
    if stype == "career_vt":
        return lambda: fetch_career_vt(week_start, week_end)
    if stype == "gobblerconnect":
        return lambda: fetch_gobblerconnect(week_start, week_end)
    if stype == "news":
        return lambda: fetch_news_vt(url)
    return None


def get_vt_data():
    """Collect this week's VT events.

    Two scrape modes are supported, set via `scraping.mode` in config.json:

    builtin        Fetch each source in sources.json with our requests +
                   BeautifulSoup parsers, producing TITLE:/WHEN:/... blocks.
                   This is the path for local (Ollama) workers, which have no
                   built-in web search.

    web_search     Hand the list of source URLs to the worker model, which uses
                   its own web-search / browsing tools to look them up itself.
                   Meant for cloud providers (Gemini, OpenAI, Claude, ...) that
                   expose a model-side search tool.

    Returns (combined_content, sources) where `sources` maps each URL to the
    text collected from it, used later to attribute events back to a link.
    """
    week_start, week_end = week_bounds()

    if SCRAPE_MODE == "web_search":
        # No local fetch -- let the (paid) worker model browse the sources.
        _stamp(f"SCRAPE  web_search mode -- worker will look up {len(SOURCES)} sources itself")
        urls = "\n".join(f"- {s.get('name','')} ({s['url']})" for s in SOURCES)
        directive = (f"You must use your web search and browsing tools to look up "
                     f"these Virginia Tech event/notice sources yourself:\n{urls}\n"
                     f"Return every event you find for the current week.")
        return directive, {}

    scrape_started = time.time()
    _stamp(f"SCRAPE  start -- collecting events for {week_start} .. {week_end}")

    collectors = []
    for source in SOURCES:
        collect = _builtin_collector(source, week_start, week_end)
        if collect:
            collectors.append((source["url"], collect))
        else:
            print(f"Skipping source with unknown type: {source!r}")

    combined_content = ""
    sources = {}
    for url, collect in collectors:
        print(f"Fetching {url}...")
        try:
            blocks = collect()
        except Exception as e:
            blocks = [f"[Error fetching {url}: {e}]"]
        # An empty source says so explicitly; it never falls back to dumping
        # navigation boilerplate, which used to flood the prompt with menu text.
        section = "\n\n".join(blocks) if blocks else "(no entries found for this week)"
        combined_content += f"\n\n=== SOURCE: {url} ===\n{section}"
        sources[url] = section
        print(f"  collected {len(blocks)} entries ({len(section)} chars)")

    _stamp(f"SCRAPE  done in {(time.time() - scrape_started) / 60:.1f} min "
           f"({len(combined_content)} chars, {count_source_events(combined_content)} event blocks)")
    return combined_content, sources

def _call_ollama(spec, prompt, thinking=None):
    if not ollama:
        raise ImportError("Ollama library not found. Run 'pip install ollama'")
    model = spec["model"]
    wait_for_model_unload(model)
    options = {'num_ctx': NUM_CTX, 'num_gpu': NUM_GPU, 'num_thread': NUM_THREAD}
    _stamp(f"LOAD    '{model}' (num_ctx={NUM_CTX}, num_gpu={NUM_GPU}, "
           f"num_thread={NUM_THREAD}, think={thinking}, prompt={len(prompt)} chars)")
    started = time.time()
    kwargs = dict(model=model, prompt=prompt, options=options, keep_alive=0)
    if thinking is not None:
        kwargs['think'] = thinking
    response = ollama.generate(**kwargs)
    _stamp(f"DONE    '{model}' returned after {(time.time() - started) / 60:.1f} min; "
           f"model unloading (keep_alive=0)")
    _log_ollama_cost(model, response)
    return response['response']


def _log_ollama_cost(model, response):
    prompt_tokens = response.get("prompt_eval_count") or 0
    gen_tokens = response.get("eval_count") or 0
    minutes = (response.get("total_duration") or 0) / 1e9 / 60
    thinking = len(response.get("thinking") or "")
    note = f", {thinking} chars of reasoning" if thinking else ""
    print(f"[{model}] {prompt_tokens} prompt tok -> {gen_tokens} generated tok"
          f"{note} in {minutes:.1f} min")


def _call_openai(spec, prompt):
    from openai import OpenAI
    client = OpenAI(base_url=spec["url"], api_key=_spec_api_key(spec))
    response = client.chat.completions.create(
        model=spec["model"],
        messages=[{"role": "user", "content": prompt}],
    )
    msg = response.choices[0].message
    return msg.content or ""


def _call_anthropic(spec, prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=_spec_api_key(spec), base_url=spec["url"])
    message = client.messages.create(
        model=spec["model"],
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _call_gemini(spec, prompt):
    import google.generativeai as genai
    genai.configure(api_key=_spec_api_key(spec))
    try:
        model = genai.GenerativeModel(model_name=spec["model"],
                                      tools=[{'google_search': {}}])
    except Exception:
        model = genai.GenerativeModel(model_name=spec["model"])
    return model.generate_content(prompt).text


def call_model(spec, prompt, thinking=None, label=""):
    """Send `prompt` to `spec`'s provider, returning the raw text reply."""
    provider = spec.get("provider", "").lower()
    tag = label or spec.get("model", provider)
    _stamp(f"CALL    provider='{provider}' model='{spec.get('model')}' "
           f"url='{spec.get('url')}' ({tag}) prompt={len(prompt)} chars")
    if provider in ("ollama", "local"):
        return _call_ollama(spec, prompt, thinking)
    if provider == "openai":
        return _call_openai(spec, prompt)
    if provider == "anthropic" or provider == "claude":
        return _call_anthropic(spec, prompt)
    if provider == "gemini":
        return _call_gemini(spec, prompt)
    raise ValueError(f"Unknown model provider: {provider!r}")


def worker_call(prompt):
    """Run the worker model on `prompt` (a get_prompt or revision prompt)."""
    return call_model(WORKER_SPEC, prompt)


def count_source_events(vt_content):
    """Exact number of event blocks in the scraped content.

    Counting is something code does perfectly and LLMs do badly, so the
    extraction-completeness check never goes to a model.
    """
    return len(re.findall(r"^TITLE: ", vt_content, re.M))


def run_judge(spec, vt_content, worker_events, today, week_end):
    """Ask one judge to REPORT issues. Judges never rewrite the event list.

    With two independent reviewers a wholesale `corrected_output` from each is
    incoherent -- whose rewrite wins? So judges only report, and the worker
    stays the single writer, re-queried once with the merged critique.

    The judge model can be any configured provider -- same or different from
    the worker.
    """
    prompt = f"""
This week runs from {today} through {week_end}.

You are a quality-control reviewer. Below is the text scraped from Virginia Tech pages, followed by a
worker AI's extracted event list. Assess BOTH completeness of extraction AND correctness of filtering.

WORKER OUTPUT:
{json.dumps(worker_events, indent=2)}

SCRAPED CONTENT:
{vt_content}

The high-interest criteria are:
{_criteria_block()}

INSTRUCTIONS:
1. Every event block in the scraped content begins with "TITLE:". Find blocks missing from "all_events".
2. Find events in "filtered_events" that match NONE of the criteria (false inclusions).
3. Find events in "all_events" that DO match a criterion but are absent from "filtered_events".
Quote titles EXACTLY as they appear. Do not rewrite the event list -- only report.

Return valid JSON only (no markdown fences, no conversational text):
{{
  "missing_events": ["exact title of an event in the scraped content but not in all_events"],
  "wrongly_included": ["exact title of an event in filtered_events matching no criterion"],
  "should_be_filtered": ["exact title of an event in all_events that matches a criterion"],
  "notes": ["any other correctness problem worth flagging"]
}}
Use empty lists when you find nothing. Do not invent events that are not in the scraped content.
"""

    model_name = spec.get("model")
    think = JUDGE_THINKING.get(model_name, True) if _is_local(spec) else None
    raw = call_model(spec, prompt, thinking=think, label=f"judge {model_name}")
    return json.loads(extract_json(raw))


def merge_judgements(reports):
    """Combine independent judge reports, weighting the two error types differently.

    The costs are asymmetric: missing a free-food event is the failure that
    actually matters, while an extra event in the email is nearly free. So
    additions (missed events, events that should have been filtered in) take the
    UNION -- one judge spotting it is enough -- while removals require UNANIMOUS
    agreement, so a single overzealous reviewer cannot prune a real hit.
    """
    def collect(key):
        merged, seen = [], set()
        for report in reports:
            for item in report.get(key) or []:
                k = _normalize_text(item).strip()
                if k and k not in seen:
                    seen.add(k)
                    merged.append(item)
        return merged

    remove_sets, labels = [], {}
    for report in reports:
        votes = set()
        for item in report.get("wrongly_included") or []:
            k = _normalize_text(item).strip()
            if k:
                votes.add(k)
                labels.setdefault(k, item)
        remove_sets.append(votes)

    agreed = set.intersection(*remove_sets) if remove_sets else set()
    flagged = set().union(*remove_sets) if remove_sets else set()

    return {
        "missing_events": collect("missing_events"),
        "should_be_filtered": collect("should_be_filtered"),
        "wrongly_included": [labels[k] for k in agreed],
        "contested_removals": [labels[k] for k in flagged - agreed],
        "notes": collect("notes"),
    }


def build_critique(verdict, expected_count, actual_count):
    """Render the merged verdict as feedback for the worker's second attempt."""
    lines = []
    if expected_count is not None and expected_count > actual_count:
        lines.append(
            f"- The scraped content contains {expected_count} event blocks but you returned "
            f"{actual_count}. Extract EVERY block that begins with 'TITLE:'."
        )
    for title in verdict["missing_events"]:
        lines.append(f"- Missing from all_events: {title}")
    for title in verdict["should_be_filtered"]:
        lines.append(f"- Matches an interest criterion, add to filtered_events: {title}")
    for title in verdict["wrongly_included"]:
        lines.append(f"- Both reviewers agree this matches no criterion, remove from "
                     f"filtered_events: {title}")
    for note in verdict["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def get_revision_prompt(vt_content, today, critique, existing_titles):
    """Ask the worker only for what is MISSING -- never for a rewrite.

    The first pass already produced a usable list, and re-emitting it is pure
    cost: generation runs at ~0.8 tok/s here, and a measured 63-event rewrite
    was 12,629 tokens and 4h17m -- over half the entire run. So the revision is
    handed the titles it already has (cheap to send, a few tokens each) and
    returns only new entries, which `apply_revision()` folds in.

    Removals are not asked for at all. A unanimous "wrongly_included" verdict is
    a title match, which code does exactly and a model does approximately.
    """
    week_end = (datetime.now() + timedelta(days=6 - datetime.now().weekday())).strftime("%A, %B %d, %Y")
    already = "\n".join(f"- {t}" for t in existing_titles) or "(none)"
    return f"""
    This week runs from {today} through {week_end}.

    You previously extracted events from the Virginia Tech content below, but the list is
    INCOMPLETE. Reviewers and an exact block count found the problems listed under FEEDBACK.

    EVENTS YOU ALREADY HAVE — do NOT repeat any of these in your answer:
{already}

    FEEDBACK:
{critique}

    Content scraped from VT:
    {vt_content}

    The high-interest criteria are:
{_criteria_block("    ")}

    YOUR TASK: return ONLY the events that are missing from the list above. Every block in the
    scraped content begins with "TITLE:" — find the ones absent from "EVENTS YOU ALREADY HAVE"
    and return them. Do not restate events you already have. Do not remove anything.

    OUTPUT FORMAT — valid JSON only, no markdown fences, no conversational text:
    {{
      "new_events": [
        {{"title": "...", "time": "...", "location": "...", "description": "...", "free_stuff": "...", "category": "..."}}
      ],
      "new_filtered": [
        {{"title": "...", "time": "...", "location": "...", "description": "...", "free_stuff": "...", "category": "...", "reason": "why it matched interest"}}
      ]
    }}
    "new_filtered" holds only those events — whether newly found or already in the list above —
    that match a criterion and are not yet marked high-interest. Use empty lists if there is
    nothing to add. Do not invent events that are not in the scraped content.
    """


def apply_revision(events, revision, verdict):
    """Fold the revision's additions into the existing list, in code.

    The worker only ever adds; merging, de-duplicating and removing all happen
    here, where they are exact. Returns (merged_events, summary string).
    """
    all_events = list(events.get("all_events") or [])
    filtered = list(events.get("filtered_events") or [])

    def key(item):
        return _normalize_text(item.get("title") if isinstance(item, dict) else item).strip()

    seen_all = {key(e) for e in all_events}
    seen_filtered = {key(e) for e in filtered}

    added_all = 0
    for event in revision.get("new_events") or []:
        k = key(event)
        if k and k not in seen_all:
            seen_all.add(k)
            all_events.append(event)
            added_all += 1

    added_filtered = 0
    for event in revision.get("new_filtered") or []:
        k = key(event)
        if k and k not in seen_filtered:
            seen_filtered.add(k)
            filtered.append(event)
            added_filtered += 1
            # A high-interest event the first pass missed entirely still belongs
            # in the full list, or the HTML report would omit what the email
            # advertises.
            if k not in seen_all:
                seen_all.add(k)
                all_events.append(event)
                added_all += 1

    # Unanimous removals, applied by exact title match rather than by asking a
    # model to reproduce the list without them.
    drop = {_normalize_text(t).strip() for t in verdict.get("wrongly_included") or []}
    drop.discard("")
    removed = 0
    if drop:
        kept = [e for e in filtered if key(e) not in drop]
        removed = len(filtered) - len(kept)
        filtered = kept

    events["all_events"] = all_events
    events["filtered_events"] = filtered
    summary = (f"+{added_all} events, +{added_filtered} high-interest, "
               f"-{removed} removed from high-interest")
    return events, summary


def get_prompt(vt_content, today):
    week_end = (datetime.now() + timedelta(days=6 - datetime.now().weekday())).strftime("%A, %B %d, %Y")
    return f"""
    This week runs from {today} through {week_end}.
    I need a comprehensive list of EVERY event, presentation, job fair, or notice happening on the Virginia Tech Blacksburg campus this week.
    
    For each event, I need:
    - Title
    - Time
    - Location
    - Description
    - Presence of Free Food or Free Items (Yes/No and what it is)
    - Category (Career, Academic, Social, etc.)

    Filter this list for "High Interest" events based on these criteria:
{_criteria_block("    ")}

    Content scraped from VT:
    {vt_content}

    OUTPUT FORMAT:
    You MUST return your response as a valid JSON object ONLY. Do not include conversational text.
    Structure:
    {{
      "all_events": [
        {{"title": "...", "time": "...", "location": "...", "description": "...", "free_stuff": "...", "category": "..."}}
      ],
      "filtered_events": [
        {{"title": "...", "time": "...", "location": "...", "description": "...", "free_stuff": "...", "category": "...", "reason": "why it matched interest"}}
      ]
    }}
    """

def strip_emojis(text):
    """Remove emoji characters from a string."""
    if text is None:
        return ""
    if isinstance(text, dict):
        parts = [f"{k}: {v}" for k, v in text.items()]
        return "; ".join(parts) if parts else str(text)
    if not isinstance(text, str):
        text = str(text)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # misc symbols
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()

def _normalize_text(text):
    """Lowercase, strip punctuation, and collapse whitespace to single spaces.

    Collapsing matters: titles reach us from HTML, from JSON, and re-typed by a
    model, so the same event can arrive as "Chess Club" or "Chess  Club". Every
    caller compares normalized strings on both sides, so this only ever makes
    matching more forgiving -- title de-duplication in the revision merge, and
    linking an event back to its source URL in the HTML report.
    """
    return " ".join(re.sub(r"[^a-z0-9\s]", "", (text or "").lower()).split())

def create_local_html(all_events, sources=None):
    today_str = datetime.now().strftime("%B %d, %Y")
    norm_sources = {url: _normalize_text(text) for url, text in (sources or {}).items()}
    fallback_url = "https://events.vt.edu/"
    html_content = f"""
    <html>
    <head>
        <title>VT Events - {today_str}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #630031; border-bottom: 2px solid #cf4420; padding-bottom: 10px; }}
            .event-card {{ border-bottom: 1px solid #ddd; padding: 15px 0; }}
            .event-card:last-child {{ border-bottom: none; }}
            .title {{ font-size: 1.2em; font-weight: bold; color: #cf4420; }}
            .meta {{ color: #555; font-size: 0.9em; margin-bottom: 5px; }}
            .free-food {{ background-color: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
            .source-link {{ color: #1565c0; font-size: 0.8em; text-decoration: none; margin-left: 8px; }}
            .source-link:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VT Campus Events - {today_str}</h1>
            <p>Total Events Found: {len(all_events)}</p>
    """
    for ev in all_events:
        free_stuff_raw = ev.get("free_stuff")
        free_stuff = strip_emojis(free_stuff_raw) or ""
        has_free_stuff = bool(free_stuff_raw) and not (
            isinstance(free_stuff_raw, str) and "no" in free_stuff_raw.lower()
        ) and not (
            isinstance(free_stuff_raw, dict) and all(
                str(v).lower().strip() in ("no", "false", "none", "") for v in free_stuff_raw.values()
            )
        )
        free_stuff_html = f'<span class="free-food">{free_stuff}</span>' if has_free_stuff else ""
        title = strip_emojis(ev.get("title")) or ""
        time_val = strip_emojis(ev.get("time")) or ""
        location = strip_emojis(ev.get("location")) or ""
        description = strip_emojis(ev.get("description")) or ""

        source_url = fallback_url
        norm_title = _normalize_text(title)
        if norm_title:
            for url, norm_section in norm_sources.items():
                if norm_title in norm_section:
                    source_url = url
                    break
        source_html = f'<a class="source-link" href="{source_url}" target="_blank">Source</a>'

        html_content += f"""
        <div class="event-card">
            <div class="title">{title} {free_stuff_html} {source_html}</div>
            <div class="meta"><strong>Time:</strong> {time_val} | <strong>Location:</strong> {location}</div>
            <div class="description">{description}</div>
        </div>
        """
    html_content += "</div></body></html>"
    with open(HTML_FILE_PATH, "w") as f: f.write(html_content)

def send_filtered_email(filtered_events):
    if not filtered_events:
        report_body = "No specific high-interest events found for today."
    else:
        report_body = "The following events matched your interests:\n\n"
        for ev in filtered_events:
            report_body += f"--- {ev.get('title') or ''} ---\n"
            report_body += f"Time: {ev.get('time') or ''} | Location: {ev.get('location') or ''}\n"
            report_body += f"Reason: {ev.get('reason') or ''}\n"
            report_body += f"Free Stuff: {ev.get('free_stuff') or ''}\n\n"

    today_str = datetime.now().strftime("%Y-%m-%d")
    message = f"Subject: VT High Interest Events - {today_str}\n\n{report_body}\nFull list: {HTML_FILE_PATH}" + (f" or at {REPORT_URL}" if REPORT_URL else "")
    
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=CONTEXT) as server:
        server.login(SENDER_EMAIL, PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.encode('utf-8'))

def extract_json(raw_response):
    raw_response = raw_response.strip()
    if "```" in raw_response:
        parts = raw_response.split("```")
        for i in range(1, len(parts)):
            candidate = parts[i].strip()
            brace_idx = candidate.find("{")
            if brace_idx != -1:
                raw_response = candidate[brace_idx:]
                break
        else:
            for i in range(len(parts) - 1, -1, -1):
                brace_idx = parts[i].find("{")
                if brace_idx != -1:
                    raw_response = parts[i][brace_idx:]
                    break
    if not raw_response.startswith("{"):
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw_response = raw_response[start:end+1]
    return raw_response




if __name__ == "__main__":
    # --dry-run exercises the whole pipeline but sends no email and writes the
    # report beside the real one, so a supervised run cannot clobber it.
    DRY_RUN = "--dry-run" in sys.argv
    if DRY_RUN:
        HTML_FILE_PATH = HTML_FILE_PATH.replace(".html", "_dryrun.html")
        print(f"*** DRY RUN: no email will be sent; report -> {HTML_FILE_PATH}")

    t0 = time.time()
    today = datetime.now().strftime("%A, %B %d, %Y")
    print("Today is "+today)
    data, sources = get_vt_data()

    week_end_str = week_bounds()[1].strftime("%A, %B %d, %Y")
    # The deterministic completeness check only applies to the builtin scrape
    # (which produces "TITLE:" blocks). In web_search mode the worker browses
    # sources itself, so there is nothing to count against.
    expected = count_source_events(data) if SCRAPE_MODE == "builtin" else None
    print(f"Scraped content contains {expected} event blocks." if expected is not None
          else "Web-search scrape mode -- no local event-block count.")

    # --- Worker phase ---
    events = None
    attempts = 0
    while events is None:
        try:
            raw_response = worker_call(get_prompt(data, today))
            raw_response = extract_json(raw_response)
            events = json.loads(raw_response)
        except Exception as e:
            attempts += 1
            print(f"Worker attempt {attempts} failed: {e}. Retrying in 5s...")
            time.sleep(5)
    print(f"[Worker] extracted {len(events.get('all_events', []))} events, "
          f"{len(events.get('filtered_events', []))} high-interest.")

    # --- Judge phase: 0, 1, or 2 independent reviewers, report-only ---
    if NUM_JUDGES > 0:
        judge_specs = [JUDGE1_SPEC]
        if NUM_JUDGES == 2:
            judge_specs.append(JUDGE2_SPEC)
        reports = []
        for spec in judge_specs:
            judge_model = spec.get("model")
            try:
                reports.append(run_judge(spec, data, events, today, week_end_str))
                print(f"[Judge {judge_model}] reported.")
            except Exception as e:
                # A judge that fails is simply absent; unanimity is then computed
                # over the judges that did respond, and the run still completes.
                print(f"[Judge {judge_model}] failed ({e}). Continuing without it.")

        if reports:
            verdict = merge_judgements(reports)
            for key in ("missing_events", "should_be_filtered", "wrongly_included"):
                for item in verdict[key]:
                    print(f"[Verdict] {key}: {item}")
            for item in verdict["contested_removals"]:
                print(f"[Verdict] contested (kept, reviewers disagreed): {item}")

            critique = build_critique(verdict, expected, len(events.get("all_events", [])))
            if critique:
                print("Reviewers found issues. Asking worker once for the missing events...")
                existing_titles = [e.get("title") for e in events.get("all_events", [])
                                   if e.get("title")]
                try:
                    raw_response = worker_call(
                        get_revision_prompt(data, today, critique, existing_titles))
                    revision = json.loads(extract_json(raw_response))
                    if revision.get("new_events") or revision.get("new_filtered"):
                        events, summary = apply_revision(events, revision, verdict)
                        print(f"[Revision] {summary} -> {len(events['all_events'])} events, "
                              f"{len(events.get('filtered_events', []))} high-interest.")
                    else:
                        print("Revision found nothing to add. Keeping original worker output.")
                except Exception as e:
                    print(f"Revision failed ({e}). Keeping original worker output.")
            else:
                print("Reviewers approved the worker output.")
        else:
            print("No judge succeeded. Proceeding with original worker output.")
    else:
        print(f"Skipping judge phase ({NUM_JUDGES} judges configured).")

    # The HTML report is written before any send-window hold, so the full list
    # is on disk and servable the moment the pipeline finishes, even while the
    # email itself is still waiting for 08:00.
    create_local_html(events.get("all_events", []), sources)
    _stamp(f"REPORT  written to {HTML_FILE_PATH}")

    filtered = events.get("filtered_events", [])
    if DRY_RUN:
        print(f"\n*** DRY RUN: skipping email. Would have sent {len(filtered)} "
              f"high-interest event(s) to {RECEIVER_EMAIL}:")
        for ev in filtered:
            print(f"  - {ev.get('title')} | {ev.get('time')} | {ev.get('reason')}")
        print(f"*** Report written to {HTML_FILE_PATH}")
        _stamp("EMAIL   dry run -- send window not applied")
    else:
        disposition = wait_for_send_window()
        _stamp(f"EMAIL   {disposition}")
        send_filtered_email(filtered)
        _stamp(f"EMAIL   sent {len(filtered)} high-interest event(s) to {RECEIVER_EMAIL}")
    elapsed = time.time() - t0
    _stamp(f"FINISH  done in {elapsed/60:.1f} min ({elapsed:.0f}s)")
