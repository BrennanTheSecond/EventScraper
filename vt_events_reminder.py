import html
import os
import re
import sys
import requests
from bs4 import BeautifulSoup
import smtplib
import ssl
from email.message import EmailMessage
import json
from dataclasses import asdict, dataclass, field
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
# Every model stage (worker, judge1, judge2) is a dict with:
#   provider : "ollama" | "gemini" | "claude" | "openai"
#   url      : the endpoint the provider is reached at (you fill this in)
#   api_key_env : name of the env var (in .env) holding this model's key
#   model    : the model name to send
#   max_tokens : optional per-stage reply cap (cloud providers)
#
# Ollama ignores api_key_env entirely -- only url + model matter.
#
# config.json itself is git-ignored (it holds real addresses and a server IP);
# config.example.json is the tracked template.
# ==========================================
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config.json")


def _json_path(base_name, config_dir):
    """Resolve a relative filename against the config file's directory."""
    if os.path.isabs(base_name):
        return base_name
    return os.path.join(config_dir, base_name)


def _normalize_criteria(raw):
    """Number the interest criteria and accept both shorthand and full form.

    A criterion is either a plain string, or an object that can also claim
    source facets outright:

        "Career / job fairs or recruiting events"
        {"text": "Offers free food", "auto_match_flags": ["FREE FOOD"]}

    Numbering is what the classification prompt and its replies use, so the
    model returns a couple of digits per event instead of restating criteria.
    """
    criteria = []
    for n, item in enumerate(raw, 1):
        if isinstance(item, str):
            criteria.append({"n": n, "text": item, "auto_match_flags": []})
        else:
            criteria.append({"n": n,
                             "text": item["text"],
                             "auto_match_flags": list(item.get("auto_match_flags") or [])})
    return criteria


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
                 f"Copy config.example.json to config.json and edit it, or point "
                 f"CONFIG_PATH / --config at another file.")

    with open(cfg_path, "r", encoding="utf-8") as fh:
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

    with open(interests_path, "r", encoding="utf-8") as fh:
        cfg["_interests"] = _normalize_criteria(json.load(fh)["criteria"])
    with open(sources_path, "r", encoding="utf-8") as fh:
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

    global MODEL_SPECS, USE_SAME_MODEL, NUM_JUDGES, MODEL_THINKING
    global SCRAPE_MODE, SOURCES
    global WORKER_SPEC, JUDGE1_SPEC, JUDGE2_SPEC
    global NUM_CTX, NUM_GPU, NUM_THREAD
    global SENDER_EMAIL, RECEIVER_EMAIL, RECIPIENTS, PASSWORD
    global HTML_FILE_PATH, REPORT_URL
    global SEND_HOUR_START, SEND_HOUR_END
    global SMTP_SERVER, SMTP_PORT
    global INTEREST_CRITERIA
    global MODEL_UNLOAD_GRACE_SECONDS, UNLOAD_POLL_SECONDS, KEEP_LOADED_SECONDS
    global HTTP_TIMEOUT, POLITE_DELAY, EVERGREEN_MAX_DAYS, INCLUDE_UNDATED
    global OUTPUT_MODE, CLASSIFY_BATCH_SIZE, CLASSIFY_DETAIL_LIMIT
    global CALL_MAX_ATTEMPTS, CALL_BACKOFF_SECONDS, CLOUD_TIMEOUT, DEFAULT_MAX_TOKENS
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
    # judge_thinking is the pre-rewrite name; both stages now read the same map,
    # because the worker is no longer special.
    MODEL_THINKING = dict(stages.get("judge_thinking") or {})
    MODEL_THINKING.update(stages.get("model_thinking") or {})
    # worker_max_attempts is the pre-rewrite name for the same thing.
    CALL_MAX_ATTEMPTS = max(1, int(stages.get("call_max_attempts",
                                              stages.get("worker_max_attempts", 3))))
    CALL_BACKOFF_SECONDS = float(stages.get("call_backoff_seconds", 5))
    CLASSIFY_BATCH_SIZE = max(1, int(stages.get("classify_batch_size", 25)))
    CLASSIFY_DETAIL_LIMIT = max(0, int(stages.get("classify_detail_limit", 400)))
    CLOUD_TIMEOUT = float(stages.get("cloud_timeout_seconds", 180))
    DEFAULT_MAX_TOKENS = int(stages.get("max_tokens", 8192))

    SCRAPE_MODE = scrape.get("mode", "builtin")
    if SCRAPE_MODE not in ("builtin",):
        sys.exit(f'scraping.mode "{SCRAPE_MODE}" is no longer supported. '
                 f'Extraction is deterministic now, so there is nothing for a '
                 f'model to browse for -- set scraping.mode to "builtin".')
    SOURCES = cfg.get("_sources", [])
    INCLUDE_UNDATED = bool(scrape.get("include_undated", True))

    # Which outputs this run produces.
    OUTPUT_MODE = str(cfg.get("output", {}).get("mode", "html+email")).lower().strip()
    OUTPUT_MODE = {"both": "html+email", "email+html": "html+email",
                   "html_only": "html", "email_only": "email"}.get(OUTPUT_MODE,
                                                                   OUTPUT_MODE)
    if OUTPUT_MODE not in ("html", "email", "html+email"):
        sys.exit(f'output.mode must be "html", "email" or "html+email" '
                 f'(got "{OUTPUT_MODE}").')

    NUM_CTX = ollama_cfg.get("num_ctx", 32768)
    NUM_GPU = ollama_cfg.get("num_gpu", 0)
    NUM_THREAD = ollama_cfg.get("num_thread", 10)
    # 0 is ideal for paid/proxied model providers (no local process to unload);
    # ~10 minutes is right for Ollama spinning large local models on/off disk.
    MODEL_UNLOAD_GRACE_SECONDS = ollama_cfg.get("model_unload_grace_seconds", 600)
    UNLOAD_POLL_SECONDS = float(ollama_cfg.get("unload_poll_seconds", 10))
    # How long Ollama holds a model between batches of the same stage.
    KEEP_LOADED_SECONDS = ollama_cfg.get("keep_loaded_seconds", "10m")

    SENDER_EMAIL = email["sender_email"]
    # receiver_email accepts a single address or a list of them.
    RECEIVER_EMAIL = email["receiver_email"]
    RECIPIENTS = ([RECEIVER_EMAIL] if isinstance(RECEIVER_EMAIL, str)
                  else list(RECEIVER_EMAIL))
    PASSWORD = os.environ.get("PASSWORD")

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


def _spec_api_key(spec):
    """Secret key for a model spec, read from .env via its api_key_env name."""
    if not spec:
        return None
    name = spec.get("api_key_env")
    if not name:
        return None
    return os.environ.get(name)


# Ollama model swap guard.
# This machine cannot hold two ~19 GB models at once. Starting to load a
# different model before the previous one has fully unloaded can crash the load,
# so a switch waits for the old model to be evicted (see wait_for_model_unload).
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
    """Wait for the previously used Ollama model to leave RAM.

    This machine cannot hold two ~19 GB models at once, so a new load must not
    begin until the old one is evicted. It used to sleep a flat
    MODEL_UNLOAD_GRACE_SECONDS (10 min) on every switch -- 30 minutes a run,
    paid whether or not the model had already gone. Since every call passes
    keep_alive=0, the model is usually gone within seconds, so poll `ollama.ps`
    instead and treat the grace as a ceiling rather than a cost.
    """
    global _last_ollama_model
    if not _last_ollama_model or _last_ollama_model == model_name:
        _last_ollama_model = model_name
        return

    previous = _last_ollama_model
    _last_ollama_model = model_name
    if MODEL_UNLOAD_GRACE_SECONDS <= 0:
        return

    _stamp(f"UNLOAD  waiting for '{previous}' to leave RAM before loading "
           f"'{model_name}' (ceiling {MODEL_UNLOAD_GRACE_SECONDS // 60} min)")
    started = time.time()
    while time.time() - started < MODEL_UNLOAD_GRACE_SECONDS:
        try:
            loaded = ollama.ps()
        except Exception as e:
            # No way to ask -- fall back to the old unconditional sleep, which
            # is the safe behavior.
            print(f"  ollama.ps() unavailable ({e}); sleeping the full grace")
            time.sleep(max(0.0, MODEL_UNLOAD_GRACE_SECONDS - (time.time() - started)))
            break
        models = getattr(loaded, "models", None)
        if models is None and isinstance(loaded, dict):
            models = loaded.get("models") or []
        names = {getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
                 for m in (models or [])}
        if previous not in names:
            break
        time.sleep(UNLOAD_POLL_SECONDS)
    _stamp(f"UNLOAD  done, '{previous}' unloaded "
           f"({(time.time() - started) / 60:.1f} min)")


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
# HTTP_TIMEOUT / POLITE_DELAY / EVERGREEN_MAX_DAYS are NOT defined here -- they
# come from config.json via apply_config(). They used to be re-declared at this
# point, which silently overrode whatever config.json said.

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


@dataclass
class Event:
    """One scraped event.

    This is the pipeline's only record type. Collectors build it, the report and
    the email read it, and the model fills in `matches`/`reason` and nothing
    else. Everything factual -- which events exist, when, where, whether the
    source itself advertises free food -- is decided in code, because the
    scraper already knows it. Asking a model to restate it was the single
    largest cost in the old pipeline and the source of every dropped event.
    """
    title: str
    when: str = ""
    dates: list = field(default_factory=list)
    location: str = ""
    description: str = ""
    category: str = ""
    host: str = ""
    link: str = ""
    source_id: str = ""
    source_url: str = ""
    # Facets the source states outright (e.g. "FREE FOOD" from a listing CSS
    # class). Authoritative -- never inferred.
    flags: list = field(default_factory=list)
    # Filled in by the classification phase: 1-based criterion numbers.
    matches: list = field(default_factory=list)
    reason: str = ""

    @property
    def key(self):
        """Normalized title, used for de-duplication and lookups."""
        return _normalize_text(self.title)

    @property
    def free_stuff(self):
        """Human-readable free-food/giveaway summary, from the source's facets."""
        return ", ".join(self.flags)


def _event_detail_text(url, limit=None):
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
    limit = EVENT_DETAIL_LIMIT if limit is None else limit
    return _strip_boilerplate(_clean(main.get_text(" ", strip=True)))[:limit]


def fetch_events_vt(source, week_start, week_end):
    """events.vt.edu -- month listing pages plus per-event detail pages."""
    events, seen = [], set()
    months = []
    for d in (week_start, week_end):
        if (d.year, d.month) not in months:
            months.append((d.year, d.month))

    for year, month in months:
        url = f"https://events.vt.edu/events/{year}/{month:02d}.html"
        try:
            soup = BeautifulSoup(_get(url), "html.parser")
        except Exception as e:
            print(f"  [error] {url}: {e}")
            continue

        for li in soup.select("li.event-page"):
            text = _clean(li.get_text(" ", strip=True))
            dates = _parse_dates(text, year)
            if not _in_week(dates, week_start, week_end):
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
            events.append(Event(
                title=title,
                when=when,
                dates=dates,
                location=", ".join(_facet_values(classes, "locations_-")),
                description=_event_detail_text(link),
                category=", ".join(_facet_values(classes, "categories_-")),
                host=", ".join(_facet_values(classes, "departments_-")),
                link=link,
                source_id=source["id"],
                source_url=source["url"],
                # Read from the listing markup, not from the detail text: the
                # site states these facets itself, so free-food detection never
                # depends on a model reading a description.
                flags=[label for cls, label in EVENTS_VT_FLAGS.items()
                       if cls in classes],
            ))
    return events


# career.vt.edu (uConnect) marks up each field inside the <li>; reading those
# nodes gives a real title. The old fallback -- splitting on a literal "Event:"
# that is not in the markup -- degraded to text[:160], producing titles like
# "Career Outfitters Pop-Up Shop: August 28th | 12:00 - 4:00 Friday, August 28,
# 2026 12pm - 4pm Fri, Aug 28 from 12pm..." for every entry.
_CAREER_TITLE_SEL = ("h1, h2, h3, h4, .event_title, .event-title, "
                     ".uc-event-title, a[href] .title")


def fetch_career_vt(source, week_start, week_end):
    """career.vt.edu -- uConnect events archive (server-rendered)."""
    url = source["url"]
    try:
        soup = BeautifulSoup(_get(url), "html.parser")
    except Exception as e:
        print(f"  [error] {url}: {e}")
        return []

    events, seen = [], set()
    for li in soup.select("li.event_item"):
        text = _clean(li.get_text(" ", strip=True))
        dates = _parse_dates(text, week_start.year)
        if not _in_week(dates, week_start, week_end):
            continue
        anchor = li.select_one("a[href]")
        link = anchor["href"] if anchor else url
        if link in seen:
            continue
        seen.add(link)

        node = li.select_one(_CAREER_TITLE_SEL)
        title = _clean(node.get_text(" ", strip=True)) if node else ""
        if not title and anchor:
            # The anchor's own text is still far better than the whole <li>.
            title = _clean(anchor.get_text(" ", strip=True))
        if not title:
            title = text[:160]
        # uConnect prefixes the heading with a screen-reader label.
        title = re.sub(r"^Event:\s*", "", title, flags=re.I)

        when = ""
        when_node = li.select_one("time, .event_date, .event-date, .uc-event-date")
        if when_node:
            when = _clean(when_node.get_text(" ", strip=True))

        events.append(Event(
            title=title[:200],
            when=when,
            dates=dates,
            description=text[:700],
            category="Career",
            link=link,
            source_id=source["id"],
            source_url=source["url"],
        ))
    return events


def fetch_gobblerconnect(source, week_start, week_end):
    """GobblerConnect -- CampusGroups JSON web service (no browser needed).

    The /events page renders client-side from this same endpoint, so we read the
    structured records directly instead of driving a headless browser.
    """
    base = "https://gobblerconnect.vt.edu/mobile_ws/v17/mobile_events_list"
    events, seen = [], set()

    for offset in range(0, GOBBLERCONNECT_MAX_EVENTS, GOBBLERCONNECT_PAGE_SIZE):
        url = (f"{base}?range={offset}&limit={GOBBLERCONNECT_PAGE_SIZE}"
               f"&filter4_contains=OR&order=undefined")
        try:
            rows = _get(url, as_json=True)
        except Exception as e:
            print(f"  [error] GobblerConnect: {e}")
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
                          .get_text(" ").replace("\u2013", "-"))
            dates = _parse_dates(when, week_start.year)
            if not _in_week(dates, week_start, week_end):
                continue
            seen.add(event_id)

            event_url = rec.get("eventUrl") or ""
            if event_url.startswith("/"):
                event_url = "https://gobblerconnect.vt.edu" + event_url
            events.append(Event(
                title=_clean(rec.get("eventName")),
                when=when,
                dates=dates,
                location=_clean(rec.get("eventLocation")),
                category=_clean(rec.get("eventCategory")),
                host=_clean(rec.get("clubName")),
                link=event_url or source["url"],
                source_id=source["id"],
                source_url=source["url"],
            ))
    return events


def fetch_news_vt(source, week_start, week_end):
    """news.vt.edu notices / stories -- link text is the headline.

    Notices carry no date in the listing markup, so unlike every other collector
    this one cannot week-filter. `scraping.include_undated` decides whether they
    are collected at all; they are marked `undated` so the report can say so.
    """
    url = source["url"]
    try:
        soup = BeautifulSoup(_get(url), "html.parser")
    except Exception as e:
        print(f"  [error] {url}: {e}")
        return []

    events, seen = [], set()
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
        dates = _parse_dates(title, week_start.year)
        if dates and not _in_week(dates, week_start, week_end):
            continue
        if not dates and not INCLUDE_UNDATED:
            continue
        events.append(Event(
            title=title,
            dates=dates,
            category="Notice",
            link=href,
            source_id=source["id"],
            source_url=source["url"],
            flags=[] if dates else ["undated"],
        ))
    return events


_COLLECTORS = {
    "events_vt": fetch_events_vt,
    "career_vt": fetch_career_vt,
    "gobblerconnect": fetch_gobblerconnect,
    "news": fetch_news_vt,
}


def render_event(event, index=None):
    """One event as a compact labelled block.

    Used for the classification prompt and for --save-scrape debugging. Only the
    fields that help a "does this match an interest" decision are included --
    the full DETAILS text is truncated hard, because the old prompts spent most
    of their ~19k tokens on description blobs the decision never needed.
    """
    head = f"[{index}] " if index is not None else ""
    lines = [f"{head}TITLE: {event.title}"]
    for label, value in (("WHEN", event.when),
                         ("LOCATION", event.location),
                         ("CATEGORY", event.category),
                         ("HOST", event.host),
                         ("FLAGS", ", ".join(event.flags))):
        if value:
            lines.append(f"    {label}: {value}")
    if event.description:
        lines.append(f"    DETAILS: {event.description[:CLASSIFY_DETAIL_LIMIT]}")
    return "\n".join(lines)


def get_vt_data():
    """Collect this week's VT events from every source in sources.json.

    Returns a flat list[Event]. Extraction is entirely deterministic: each
    source has a dedicated parser, and what those parsers produce IS the event
    list. There is no model in this path, so events cannot be dropped,
    duplicated or invented, and the completeness checking, judging and revision
    machinery that used to police a model's transcription is gone with it.
    """
    week_start, week_end = week_bounds()
    scrape_started = time.time()
    _stamp(f"SCRAPE  start -- collecting events for {week_start} .. {week_end}")

    events = []
    for source in SOURCES:
        collect = _COLLECTORS.get(source.get("type"))
        if not collect:
            print(f"Skipping source with unknown type: {source!r}")
            continue
        print(f"Fetching {source['url']}...")
        try:
            found = collect(source, week_start, week_end)
        except Exception as e:
            print(f"  [error] {source['url']}: {e}")
            found = []
        print(f"  collected {len(found)} events")
        if not found:
            # A source that used to return events and now returns none is the
            # silent failure mode a VT redesign causes, so say so loudly rather
            # than letting an empty digest look like a quiet week.
            print(f"  WARNING: {source['id']} returned no events this week")
        events.extend(found)

    events = dedupe_events(events)
    _stamp(f"SCRAPE  done in {(time.time() - scrape_started) / 60:.1f} min "
           f"({len(events)} events after de-duplication)")
    return events


def dedupe_events(events):
    """Drop repeats of the same title, keeping the richest record.

    The same event is often listed on both events.vt.edu and GobblerConnect.
    Preferring the entry with the most filled-in fields keeps whichever source
    carried the description and the facets.
    """
    best = {}
    order = []
    for event in events:
        k = event.key
        if not k:
            continue
        if k not in best:
            best[k] = event
            order.append(k)
            continue
        incumbent = best[k]
        if _richness(event) > _richness(incumbent):
            # Keep any flags the loser had -- facets are additive evidence.
            for flag in incumbent.flags:
                if flag not in event.flags:
                    event.flags.append(flag)
            best[k] = event
        else:
            for flag in event.flags:
                if flag not in incumbent.flags:
                    incumbent.flags.append(flag)
    return [best[k] for k in order]


def _richness(event):
    return sum(bool(v) for v in (event.when, event.location, event.description,
                                 event.category, event.host, event.link,
                                 event.flags))


# ---------------------------------------------------------------------------
# Providers
#
# One interface, one implementation per backend. Local and cloud are both first
# class and optimized for opposite things: Ollama for an overnight box where one
# ~19 GB model must leave RAM before the next loads, cloud for someone who wants
# the digest in five minutes and can run every batch at once.
# ---------------------------------------------------------------------------


class Provider:
    """Base interface. `generate` returns raw text; JSON handling is per-backend."""

    #: Cloud backends can run batches in parallel; a single local llama.cpp
    #: process cannot, and trying would just thrash the same memory bandwidth.
    supports_concurrency = False
    #: Only Ollama has a model to evict before the next one loads.
    needs_unload_guard = False

    def __init__(self, spec):
        self.spec = spec
        self.model = spec.get("model", "")
        self.url = spec.get("url") or None
        self.max_tokens = int(spec.get("max_tokens") or DEFAULT_MAX_TOKENS)

    def generate(self, prompt, want_json=False, thinking=None, keep_loaded=False):
        raise NotImplementedError

    def __str__(self):
        return f"{self.spec.get('provider', '?')}:{self.model}"


# Models that turned out to have no reasoning mode. Ollama answers a request
# carrying `think` for such a model with a 400 rather than ignoring the
# parameter, so the first call for a given model discovers this and every later
# one drops `think` up front instead of paying the failure again.
_NO_THINKING_MODELS = set()
_NO_THINKING_RE = re.compile(r"does not support thinking", re.I)


class OllamaProvider(Provider):
    needs_unload_guard = True

    def generate(self, prompt, want_json=False, thinking=None, keep_loaded=False):
        if not ollama:
            raise ImportError("Ollama library not found. Run 'pip install ollama'")
        if thinking is not None and self.model in _NO_THINKING_MODELS:
            # Already known to have no reasoning mode -- run it plainly.
            thinking = None
        wait_for_model_unload(self.model)
        options = {"num_ctx": NUM_CTX, "num_gpu": NUM_GPU, "num_thread": NUM_THREAD}
        if want_json:
            # Ollama constrains sampling to valid JSON, which removes most of
            # what extract_json() used to have to repair.
            options["format"] = "json"
        _stamp(f"LOAD    '{self.model}' (num_ctx={NUM_CTX}, num_gpu={NUM_GPU}, "
               f"num_thread={NUM_THREAD}, think={thinking}, "
               f"prompt={len(prompt)} chars)")
        started = time.time()
        # keep_alive=0 evicts the model the moment the call returns. That is
        # right between stages, but wrong between batches of the same stage:
        # measured here, a cold load costs 393 s of the 923 s batch (prefill
        # runs at 5.95 tok/s while 19 GB faults in from the SATA SSD, against
        # 1.25 tok/s of actual decode). Paying that once per batch instead of
        # once per stage would add ~13 min per reviewer for nothing.
        keep_alive = KEEP_LOADED_SECONDS if keep_loaded else 0
        kwargs = dict(model=self.model, prompt=prompt, options=options,
                      keep_alive=keep_alive)
        if want_json:
            kwargs["format"] = "json"
        if thinking is not None:
            kwargs["think"] = thinking
        try:
            response = ollama.generate(**kwargs)
        except Exception as e:
            # A model with no reasoning mode is a fine model to use -- it just
            # cannot be asked to reason. Drop `think` and run it, rather than
            # failing the batch over a parameter the work never needed.
            if thinking is None or not _NO_THINKING_RE.search(str(e)):
                raise
            _NO_THINKING_MODELS.add(self.model)
            kwargs.pop("think", None)
            _stamp(f"LOAD    '{self.model}' has no reasoning mode -- "
                   f"re-running it without thinking")
            response = ollama.generate(**kwargs)
        _stamp(f"DONE    '{self.model}' returned after "
               f"{(time.time() - started) / 60:.1f} min"
               + ("; staying loaded for the next batch" if keep_loaded
                  else "; model unloading (keep_alive=0)"))
        _log_ollama_cost(self.model, response)
        return response["response"]


class OpenAIProvider(Provider):
    supports_concurrency = True

    def generate(self, prompt, want_json=False, thinking=None, keep_loaded=False):
        from openai import OpenAI
        client = OpenAI(base_url=self.url, api_key=_spec_api_key(self.spec),
                        timeout=CLOUD_TIMEOUT)
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class AnthropicProvider(Provider):
    supports_concurrency = True

    def generate(self, prompt, want_json=False, thinking=None, keep_loaded=False):
        import anthropic
        client = anthropic.Anthropic(api_key=_spec_api_key(self.spec),
                                     base_url=self.url, timeout=CLOUD_TIMEOUT)
        messages = [{"role": "user", "content": prompt}]
        if want_json:
            # Prefilling the assistant turn with "{" is Anthropic's JSON mode:
            # the reply continues from there, so it cannot open with prose.
            messages.append({"role": "assistant", "content": "{"})
        message = client.messages.create(model=self.model,
                                         max_tokens=self.max_tokens,
                                         messages=messages)
        text = "".join(b.text for b in message.content if b.type == "text")
        return ("{" + text) if want_json else text


class GeminiProvider(Provider):
    supports_concurrency = True

    def generate(self, prompt, want_json=False, thinking=None, keep_loaded=False):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=_spec_api_key(self.spec))
        config = types.GenerateContentConfig(max_output_tokens=self.max_tokens)
        if want_json:
            config.response_mime_type = "application/json"
        response = client.models.generate_content(model=self.model,
                                                  contents=prompt,
                                                  config=config)
        return response.text or ""


_PROVIDERS = {
    "ollama": OllamaProvider,
    "local": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "gemini": GeminiProvider,
}


def make_provider(spec):
    name = (spec.get("provider") or "").lower()
    try:
        return _PROVIDERS[name](spec)
    except KeyError:
        raise ValueError(f"Unknown model provider: {spec.get('provider')!r}. "
                         f"Known: {', '.join(sorted(set(_PROVIDERS)))}")


def _log_ollama_cost(model, response):
    prompt_tokens = response.get("prompt_eval_count") or 0
    gen_tokens = response.get("eval_count") or 0
    minutes = (response.get("total_duration") or 0) / 1e9 / 60
    thinking = len(response.get("thinking") or "")
    note = f", {thinking} chars of reasoning" if thinking else ""
    print(f"[{model}] {prompt_tokens} prompt tok -> {gen_tokens} generated tok"
          f"{note} in {minutes:.1f} min")


# Errors that a retry cannot fix: a missing model, a bad key, a refused
# request. Retrying these only delays the run and muddies the log.
#
# "does not support thinking" is deliberately NOT here: OllamaProvider handles
# it by re-running the model without `think`, so it never reaches this point.
_PERMANENT_ERROR_RE = re.compile(
    r"not found|invalid[_ ]api[_ ]key|authentication|unauthorized|"
    r"permission denied|model .* does not exist|does not exist", re.I)


def _is_permanent(error):
    return bool(_PERMANENT_ERROR_RE.search(str(error)))


def call_model(provider, prompt, want_json=True, thinking=None, label="",
               keep_loaded=False):
    """One model call with retry/backoff, returning raw text.

    Retries cover the transient cloud failures (429s, 5xx, dropped
    connections) that used to kill a whole stage, and the occasional local
    hiccup. The caller decides what to do when every attempt fails.
    """
    tag = label or str(provider)
    last_error = None
    for attempt in range(1, CALL_MAX_ATTEMPTS + 1):
        try:
            _stamp(f"CALL    {provider} ({tag}) attempt {attempt}/"
                   f"{CALL_MAX_ATTEMPTS}, prompt={len(prompt)} chars")
            return provider.generate(prompt, want_json=want_json,
                                     thinking=thinking, keep_loaded=keep_loaded)
        except Exception as e:
            last_error = e
            print(f"  call failed ({type(e).__name__}: {e})")
            if _is_permanent(e):
                print("  not retrying -- this error will not resolve on a retry")
                break
            if attempt < CALL_MAX_ATTEMPTS:
                backoff = CALL_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"  retrying in {backoff}s")
                time.sleep(backoff)
    raise RuntimeError(f"{tag}: all {CALL_MAX_ATTEMPTS} attempts failed: {last_error}")


# ---------------------------------------------------------------------------
# Classification
#
# The only thing a model is actually needed for. Extraction is deterministic, so
# what remains is one genuinely fuzzy question per event -- "does this match an
# interest criterion?" -- answered a batch at a time, returning an index and a
# short list of criterion numbers rather than a restatement of the event.
# ---------------------------------------------------------------------------


def _criteria_block(indent="   "):
    return "\n".join(f"{indent}{c['n']}. {c['text']}" for c in INTEREST_CRITERIA)


def auto_matches(event):
    """Criteria the source itself already settles, with no model call.

    events.vt.edu states free food and giveaways as listing CSS classes. A
    criterion can claim those facets via `auto_match_flags` in interests.json,
    and then an event carrying the facet matches outright -- the fact is
    published by the source, so inferring it would only add a way to be wrong.
    """
    matched = []
    flags = {f.upper() for f in event.flags}
    for criterion in INTEREST_CRITERIA:
        wanted = {f.upper() for f in criterion.get("auto_match_flags") or []}
        if wanted & flags:
            matched.append(criterion["n"])
    return matched


def get_classify_prompt(batch, start_index, today, week_end):
    """Prompt for one batch. Output is a few tokens per event."""
    listing = "\n".join(render_event(e, start_index + i)
                         for i, e in enumerate(batch))
    return f"""This week runs from {today} through {week_end}.

You are filtering Virginia Tech campus events for one person. For each numbered
event below, decide which of their interest criteria it matches.

INTEREST CRITERIA:
{_criteria_block()}

EVENTS:
{listing}

Return valid JSON only -- no markdown fences, no conversational text:
{{"results": [{{"i": <the number in brackets>, "matches": [<criterion numbers>], "reason": "<short phrase>"}}]}}

Rules:
- Include exactly one entry for EVERY event shown, in order.
- "matches" is a list of criterion numbers from the list above. Use [] when the
  event matches none of them.
- "reason" is one short phrase naming what matched; use "" when "matches" is [].
- A FLAGS line comes from the event listing itself and is reliable evidence.
- Judge only the event shown. Do not invent events.
"""


def _parse_classification(raw, batch_size, start_index):
    """Turn one batch reply into {index: (matches, reason)}."""
    payload = json.loads(extract_json(raw))
    results = payload.get("results")
    if results is None and isinstance(payload, dict):
        # Some models return a bare list under another key, or the list itself.
        for value in payload.values():
            if isinstance(value, list):
                results = value
                break
    parsed = {}
    valid = {c["n"] for c in INTEREST_CRITERIA}
    for entry in results or []:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("i"))
        except (TypeError, ValueError):
            continue
        if not (start_index <= index < start_index + batch_size):
            continue
        matches = [int(m) for m in (entry.get("matches") or [])
                   if str(m).strip().lstrip("-").isdigit() and int(m) in valid]
        parsed[index] = (matches, str(entry.get("reason") or "").strip())
    return parsed


def classify(events, provider, today, week_end, label):
    """Classify every event with one model, a batch at a time.

    Returns {index: (matches, reason)}. A batch whose reply never parses is
    skipped rather than aborting the run -- the cost of a bad batch is 25
    events falling back to whatever the other reviewers said, not the loss of
    the whole night's work.
    """
    results = {}
    total = (len(events) + CLASSIFY_BATCH_SIZE - 1) // CLASSIFY_BATCH_SIZE
    for batch_no, start in enumerate(range(0, len(events), CLASSIFY_BATCH_SIZE), 1):
        batch = events[start:start + CLASSIFY_BATCH_SIZE]
        prompt = get_classify_prompt(batch, start, today, week_end)
        tag = f"{label} batch {batch_no}/{total}"
        try:
            raw = call_model(provider, prompt, want_json=True,
                             thinking=_thinking_for(provider), label=tag,
                             # Hold the model in RAM until this reviewer's last
                             # batch, then let it go so the next stage can load.
                             keep_loaded=batch_no < total)
            parsed = _parse_classification(raw, len(batch), start)
        except Exception as e:
            print(f"[{tag}] failed ({e}). Skipping this batch.")
            continue
        missing = len(batch) - len(parsed)
        if missing:
            print(f"[{tag}] returned {len(parsed)}/{len(batch)} verdicts "
                  f"({missing} event(s) left unjudged by this reviewer)")
        results.update(parsed)
    return results


def _thinking_for(provider):
    """Ollama reasoning on/off for this model, or None to leave it to Ollama.

    Returning None matters: it means `think` is never sent, which is the only
    thing that works for a model with no reasoning mode -- Ollama rejects the
    request outright with "does not support thinking". So a model absent from
    the map keeps Ollama's own default rather than being forced either way.
    """
    if not isinstance(provider, OllamaProvider):
        return None
    return MODEL_THINKING.get(provider.model)


def run_classification(events, today, week_end):
    """Fill in `matches`/`reason` on every event.

    Reviewers vote by union: an event is high-interest if the worker OR any
    judge says it matches. That keeps the asymmetry the old merge rules were
    built for -- a missed free-food event is the failure that matters, an extra
    line in the email costs nothing -- without any of the merge machinery,
    because there is no longer a list for reviewers to disagree about.
    """
    for event in events:
        event.matches = auto_matches(event)
        if event.matches:
            event.reason = "flagged by the source: " + event.free_stuff

    specs = [("worker", WORKER_SPEC)]
    if NUM_JUDGES >= 1:
        specs.append(("judge1", JUDGE1_SPEC))
    if NUM_JUDGES >= 2:
        specs.append(("judge2", JUDGE2_SPEC))

    reviewed = 0
    for label, spec in specs:
        try:
            provider = make_provider(spec)
        except Exception as e:
            print(f"[{label}] unusable ({e}). Skipping this reviewer.")
            continue
        try:
            verdicts = classify(events, provider, today, week_end, label)
        except Exception as e:
            print(f"[{label}] failed ({e}). Continuing without it.")
            continue
        reviewed += 1
        added = 0
        for index, (matches, reason) in verdicts.items():
            event = events[index]
            new = [m for m in matches if m not in event.matches]
            if new:
                event.matches.extend(new)
                added += 1
                if not event.reason and reason:
                    event.reason = reason
            elif matches and not event.reason and reason:
                event.reason = reason
        print(f"[{label}] added {added} new high-interest match(es)")

    if not reviewed:
        print("WARNING: no reviewer succeeded. High-interest events are those "
              "the sources flagged themselves.")
    return reviewed


def high_interest(events):
    return [e for e in events if e.matches]


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

def _e(value):
    """Escape for HTML, after stripping emojis. Never interpolate raw text.

    Everything rendered here is either scraped from the web or written by a
    model, so it is untrusted: an unescaped '&' mangles the page and an
    unescaped tag in a title would inject script into the served report.
    """
    return html.escape(strip_emojis(value) or "", quote=True)


_DAY_FMT = "%A, %B %d"


def group_by_day(events, week_start, week_end):
    """Group events into (heading, [events]) pairs, in calendar order.

    Possible only because `dates` is parsed by the scraper rather than restated
    as free text by a model. Anything without a usable date -- news notices --
    lands in a trailing group that says so.
    """
    buckets = {}
    ongoing, undated = [], []
    for event in events:
        in_week = [d for d in event.dates if week_start <= d <= week_end]
        if in_week:
            buckets.setdefault(min(in_week), []).append(event)
        elif event.dates:
            # A run that starts before Monday and ends after Sunday. Filing it
            # under its start date would date the report to a week it is not in.
            ongoing.append(event)
        else:
            undated.append(event)
    groups = [(day.strftime(_DAY_FMT), buckets[day]) for day in sorted(buckets)]
    if ongoing:
        groups.append(("Ongoing this week", ongoing))
    if undated:
        groups.append(("Undated notices", undated))
    return groups


def create_local_html(events, week_start, week_end, path):
    today_str = datetime.now().strftime("%B %d, %Y")
    starred = high_interest(events)
    criteria_by_n = {c["n"]: c["text"] for c in INTEREST_CRITERIA}

    parts = [f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VT Events - {_e(today_str)}</title>
<style>
  :root {{ color-scheme: light dark; --ground:#f4f2f2; --card:#fff; --ink:#1a1315;
           --ink2:#5f5257; --rule:#e0d6d8; --maroon:#630031; --orange:#cf4420;
           --good:#2c6a4e; --goodbg:#e8f5ee; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ground:#151011; --card:#1d1719; --ink:#efe8ea; --ink2:#a4939a;
             --rule:#352a2d; --maroon:#db8fae; --orange:#f0834f;
             --good:#6fc79b; --goodbg:#1d2f27; }}
  }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin:0;
          background: var(--ground); color: var(--ink); line-height:1.55; }}
  .container {{ max-width: 900px; margin: 0 auto; padding: 24px 20px 64px; }}
  h1 {{ color: var(--maroon); border-bottom: 2px solid var(--orange);
        padding-bottom: 10px; margin: 0 0 6px; font-size: 1.7rem; }}
  .summary {{ color: var(--ink2); margin: 0 0 28px; font-size: .95rem; }}
  h2 {{ font-size: 1rem; text-transform: uppercase; letter-spacing: .08em;
        color: var(--ink2); margin: 34px 0 10px; font-weight: 600; }}
  .event-card {{ background: var(--card); border: 1px solid var(--rule);
                 border-radius: 6px; padding: 14px 16px; margin-bottom: 10px; }}
  .event-card.high {{ border-left: 4px solid var(--maroon); }}
  .title {{ font-size: 1.05rem; font-weight: 600; color: var(--orange); }}
  .title a {{ color: inherit; text-decoration: none; }}
  .title a:hover, .title a:focus-visible {{ text-decoration: underline; }}
  .meta {{ color: var(--ink2); font-size: .87rem; margin-top: 2px; }}
  .description {{ font-size: .92rem; margin-top: 6px; }}
  .tag {{ display: inline-block; font-size: .75rem; font-weight: 600;
          border-radius: 4px; padding: 1px 7px; margin-left: 6px;
          vertical-align: middle; }}
  .tag.free {{ background: var(--goodbg); color: var(--good); }}
  .tag.high {{ background: var(--maroon); color: var(--ground); }}
  .why {{ font-size: .84rem; color: var(--ink2); margin-top: 5px; font-style: italic; }}
</style>
</head>
<body>
<div class="container">
<h1>VT Campus Events - {_e(today_str)}</h1>
<p class="summary">{len(events)} events for
{_e(week_start.strftime("%B %d"))} - {_e(week_end.strftime("%B %d, %Y"))} &middot;
{len(starred)} match your interests</p>
"""]

    for heading, group in group_by_day(events, week_start, week_end):
        parts.append(f'<h2>{_e(heading)}</h2>\n')
        for ev in group:
            classes = "event-card high" if ev.matches else "event-card"
            title = _e(ev.title)
            if ev.link:
                title = (f'<a href="{html.escape(ev.link, quote=True)}" '
                         f'target="_blank" rel="noopener">{title}</a>')
            tags = ""
            if ev.matches:
                why = "; ".join(criteria_by_n.get(n, str(n)) for n in ev.matches)
                tags += f'<span class="tag high">{_e(why)}</span>'
            if ev.free_stuff and ev.free_stuff != "undated":
                tags += f'<span class="tag free">{_e(ev.free_stuff)}</span>'
            meta = " &middot; ".join(_e(v) for v in (ev.when, ev.location, ev.host) if v)
            parts.append(f'<div class="{classes}">'
                         f'<div class="title">{title}{tags}</div>'
                         + (f'<div class="meta">{meta}</div>' if meta else "")
                         + (f'<div class="description">{_e(ev.description[:400])}</div>'
                            if ev.description else "")
                         + (f'<div class="why">{_e(ev.reason)}</div>' if ev.reason else "")
                         + "</div>\n")

    parts.append("</div>\n</body>\n</html>\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))


def _email_bodies(starred, report_link):
    """Render the digest as (plain text, html) for a multipart/alternative send."""
    criteria_by_n = {c["n"]: c["text"] for c in INTEREST_CRITERIA}
    if not starred:
        text = "No high-interest events found this week."
        body_html = "<p>No high-interest events found this week.</p>"
    else:
        lines = ["The following events matched your interests:", ""]
        cards = []
        for ev in starred:
            why = ev.reason or "; ".join(criteria_by_n.get(n, str(n)) for n in ev.matches)
            title = strip_emojis(ev.title)
            when = strip_emojis(ev.when)
            location = strip_emojis(ev.location)
            lines += [f"--- {title} ---",
                      f"When: {when or 'see listing'} | Where: {location or 'see listing'}",
                      f"Why: {strip_emojis(why)}"]
            if ev.free_stuff:
                lines.append(f"Free: {strip_emojis(ev.free_stuff)}")
            if ev.link:
                lines.append(ev.link)
            lines.append("")
            heading = (f'<a href="{html.escape(ev.link, quote=True)}" '
                       f'style="color:#cf4420;text-decoration:none">{_e(ev.title)}</a>'
                       if ev.link else _e(ev.title))
            cards.append(
                '<div style="margin:0 0 18px 0">'
                f'<div style="font-weight:600;font-size:16px">{heading}</div>'
                + (f'<div style="color:#555;font-size:14px">'
                   f'{_e(when)}{" &middot; " if when and location else ""}'
                   f'{_e(location)}</div>' if (when or location) else "")
                + f'<div style="font-size:14px">{_e(why)}</div>'
                + (f'<div style="font-size:14px;color:#2c6a4e"><strong>Free:</strong> '
                   f'{_e(ev.free_stuff)}</div>' if ev.free_stuff else "")
                + '</div>')
        text = "\n".join(lines)
        body_html = ("<p>The following events matched your interests:</p>"
                     + "".join(cards))

    if report_link:
        text += f"\n\nFull list: {report_link}"
        if report_link.startswith("http"):
            link_html = (f'<p><a href="{html.escape(report_link, quote=True)}">'
                         f'Full list of events</a></p>')
        else:
            link_html = f"<p>Full list: {_e(report_link)}</p>"
    else:
        link_html = ""
    body_html = (f'<html><body style="font-family:system-ui,sans-serif">'
                 f'{body_html}{link_html}</body></html>')
    return text, body_html


def send_filtered_email(starred, report_link):
    """Send the digest as a proper MIME message.

    This used to be a hand-built "Subject: ...\n\n..." string with no headers
    and no declared charset, so any non-ASCII character in a VT event title
    reached the client as mojibake and the message had no To:/From:.
    """
    text, body_html = _email_bodies(starred, report_link)

    message = EmailMessage()
    message["Subject"] = f"VT High Interest Events - {datetime.now():%Y-%m-%d}"
    message["From"] = SENDER_EMAIL
    message["To"] = ", ".join(RECIPIENTS)
    message.set_content(text)
    message.add_alternative(body_html, subtype="html")

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=CONTEXT) as server:
        server.login(SENDER_EMAIL, PASSWORD)
        server.send_message(message)


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




def _flag_value(name, default=None):
    """Value of a `--flag value` command-line argument."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def save_scrape(path, events):
    """Write a scrape to disk so later runs can replay it without refetching."""
    payload = {"saved_at": datetime.now().isoformat(),
               "events": [_event_to_dict(e) for e in events]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _stamp(f"SCRAPE  saved {len(events)} events to {path} (replay with --from-scrape)")


def _event_to_dict(event):
    data = asdict(event)
    data["dates"] = [d.isoformat() for d in event.dates]
    return data


def _event_from_dict(data):
    data = dict(data)
    data["dates"] = [date.fromisoformat(d) for d in data.get("dates") or []]
    return Event(**data)


def load_scrape(path):
    """Replay a saved scrape. Iterating on prompts or the report otherwise costs
    a full refetch; combined with --no-llm it makes a change testable in
    seconds instead of hours."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    events = [_event_from_dict(d) for d in payload["events"]]
    _stamp(f"SCRAPE  replayed {len(events)} events from {path} "
           f"(captured {payload.get('saved_at')})")
    return events


def main():
    argv = sys.argv
    dry_run = "--dry-run" in argv
    no_llm = "--no-llm" in argv
    from_scrape = _flag_value("--from-scrape")
    save_path = _flag_value("--save-scrape")
    if no_llm:
        dry_run = True

    want_html = OUTPUT_MODE in ("html", "html+email")
    want_email = OUTPUT_MODE in ("email", "html+email") and not dry_run

    html_path = HTML_FILE_PATH
    if dry_run:
        html_path = html_path.replace(".html", "_dryrun.html")
        print(f"*** DRY RUN: no email will be sent"
              + (f"; report -> {html_path}" if want_html else ""))

    # A missing password used to surface only at send time, hours after the work
    # was done. Fail before spending any of it.
    if want_email and not PASSWORD:
        sys.exit("PASSWORD is not set in .env -- the email would fail after the "
                 "whole pipeline ran. Set it, or use --dry-run, or set "
                 "output.mode to \"html\".")

    t0 = time.time()
    today = datetime.now().strftime("%A, %B %d, %Y")
    week_start, week_end = week_bounds()
    print(f"Today is {today}")
    print(f"Output mode: {OUTPUT_MODE}"
          + ("" if want_email else " (no email this run)"))

    events = load_scrape(from_scrape) if from_scrape else get_vt_data()
    if save_path:
        save_scrape(save_path, events)
    if not events:
        print("No events collected. Nothing to report.")

    # --- Classification ---
    if no_llm:
        for event in events:
            event.matches = auto_matches(event)
            if event.matches:
                event.reason = "flagged by the source: " + event.free_stuff
        print(f"*** --no-llm: no model called; {len(high_interest(events))} "
              f"high-interest from source flags alone.")
    else:
        run_classification(events, today, week_end.strftime("%A, %B %d, %Y"))

    starred = high_interest(events)
    print(f"[Result] {len(events)} events, {len(starred)} high-interest.")

    # The report is written before any send-window hold, so the full list is on
    # disk and servable the moment the pipeline finishes, even while the email
    # is still waiting for 08:00.
    report_link = ""
    if want_html:
        create_local_html(events, week_start, week_end, html_path)
        _stamp(f"REPORT  written to {html_path}")
        report_link = REPORT_URL or html_path
    else:
        _stamp("REPORT  skipped (output.mode does not include html)")

    if want_email:
        disposition = wait_for_send_window()
        _stamp(f"EMAIL   {disposition}")
        send_filtered_email(starred, report_link)
        _stamp(f"EMAIL   sent {len(starred)} high-interest event(s) to "
               f"{', '.join(RECIPIENTS)}")
    elif dry_run and OUTPUT_MODE in ("email", "html+email"):
        print(f"\n*** DRY RUN: skipping email. Would have sent {len(starred)} "
              f"high-interest event(s) to {', '.join(RECIPIENTS)}:")
        for ev in starred:
            print(f"  - {ev.title} | {ev.when} | {ev.reason}")
        _stamp("EMAIL   dry run -- send window not applied")
    else:
        _stamp("EMAIL   skipped (output.mode does not include email)")

    elapsed = time.time() - t0
    _stamp(f"FINISH  done in {elapsed/60:.1f} min ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
