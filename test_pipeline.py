"""Checks for the parts of the pipeline that are pure functions.

No pytest, no fixtures, no network: `python test_pipeline.py` and read the
output. These cover the code that turns untrusted input -- a model's reply, a
scraped page -- into decisions, which is where a silent regression would cost a
whole overnight run before anyone noticed.

Everything that needs a live site or a live model is exercised instead by:

    python vt_events_reminder.py --dry-run --save-scrape scrape.json
    python vt_events_reminder.py --no-llm --from-scrape scrape.json
"""

import importlib.util
import sys
from datetime import date

# The script reads config at import, so give it a clean argv.
_argv, sys.argv = sys.argv, ["test_pipeline"]
_spec = importlib.util.spec_from_file_location("vt", "vt_events_reminder.py")
vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vt)
sys.argv = _argv

# These checks must not depend on whatever the user currently has in
# interests.json -- editing your own interests should never fail the suite.
vt.INTEREST_CRITERIA = vt._normalize_criteria([
    {"text": "Offers free food or free items",
     "auto_match_flags": ["FREE FOOD", "FREE ITEMS / GIVEAWAYS"]},
    "Career / job fairs or recruiting events",
    "Events related to Fencing (sport)",
    "Events related to Data Science, AI, or Machine Learning",
])

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ""))
        FAILURES.append(label)


def section(name):
    print(f"\n{name}")


def test_parse_classification():
    section("_parse_classification -- a model reply is untrusted input")
    p = vt._parse_classification

    check("well-formed reply",
          p('{"results":[{"i":0,"matches":[1],"reason":"pizza"},'
            '{"i":1,"matches":[],"reason":""}]}', 2, 0)
          == {0: ([1], "pizza"), 1: ([], "")})

    check("markdown fences and conversational preamble are stripped",
          p('Sure!\n```json\n{"results":[{"i":5,"matches":[2],"reason":"x"}]}\n```', 3, 5)
          == {5: ([2], "x")})

    check("an index outside this batch is dropped",
          p('{"results":[{"i":99,"matches":[1]},{"i":0,"matches":[2]}]}', 2, 0)
          == {0: ([2], "")})

    check("a criterion number that does not exist is dropped",
          p('{"results":[{"i":0,"matches":[1,17,"3"]}]}', 1, 0) == {0: ([1, 3], "")},
          "and a digit sent as a string is still accepted")

    check("a results list under an unexpected key is still found",
          p('{"events":[{"i":0,"matches":[4],"reason":"ml"}]}', 1, 0) == {0: ([4], "ml")})

    check("malformed entries are skipped rather than fatal",
          p('{"results":["nope",{"i":"x"},{"i":0,"matches":[1]}]}', 1, 0) == {0: ([1], "")})

    try:
        p("not json at all", 1, 0)
        check("unparseable reply raises so the batch is skipped", False)
    except Exception:
        check("unparseable reply raises so the batch is skipped", True)


def test_dedupe():
    section("dedupe_events -- the same event appears on several sources")
    sparse = vt.Event(title="Free Pizza Night", link="a", flags=["FREE FOOD"])
    rich = vt.Event(title="free  pizza night!", link="b", location="Squires",
                    description="d", when="w", category="c")
    out = vt.dedupe_events([sparse, rich])
    check("titles differing only by case, spacing and punctuation collapse", len(out) == 1)
    check("the record with more fields wins", out and out[0].link == "b")
    check("flags from the discarded record are kept",
          out and out[0].flags == ["FREE FOOD"],
          "facets are additive evidence -- losing one loses a free-food match")
    check("an untitled record is dropped", vt.dedupe_events([vt.Event(title="")]) == [])


def test_auto_matches():
    section("auto_matches -- facets the source states outright, no model call")
    check("a FREE FOOD facet matches the criterion that claims it",
          vt.auto_matches(vt.Event(title="x", flags=["FREE FOOD"])) == [1])
    check("a giveaway facet matches it too",
          vt.auto_matches(vt.Event(title="x", flags=["FREE ITEMS / GIVEAWAYS"])) == [1])
    check("an unflagged event matches nothing automatically",
          vt.auto_matches(vt.Event(title="x")) == [])
    check("an unclaimed facet does not match",
          vt.auto_matches(vt.Event(title="x", flags=["free parking"])) == [])


def test_permanent_errors():
    section("_is_permanent -- do not burn retries on an error that cannot resolve")
    check("a bad API key", vt._is_permanent(Exception("invalid_api_key")))
    check("an unknown model", vt._is_permanent(Exception("model foo not found")))
    check("a rate limit IS retried", not vt._is_permanent(Exception("429 rate limit")))
    check("a timeout IS retried", not vt._is_permanent(Exception("Read timed out")))
    check("a model with no reasoning mode is NOT treated as permanent",
          not vt._is_permanent(Exception('"granite4.1:8b" does not support thinking')),
          "OllamaProvider re-runs it without `think`, so it never gets this far")


def test_thinking_fallback():
    section("no reasoning mode -- run the model anyway, without thinking")
    check("Ollama's rejection is recognised",
          bool(vt._NO_THINKING_RE.search('"granite4.1:8b" does not support thinking')))
    check("an unrelated failure is not mistaken for it",
          not vt._NO_THINKING_RE.search("connection refused"))

    calls = []

    class FakeOllama:
        """Rejects `think` once, the way Ollama does, then succeeds."""
        def generate(self, **kwargs):
            calls.append(dict(kwargs))
            if "think" in kwargs:
                raise RuntimeError('"fake:1b" does not support thinking')
            return {"response": '{"ok":1}', "prompt_eval_count": 1, "eval_count": 1}

    real_ollama, vt.ollama = vt.ollama, FakeOllama()
    remembered = set(vt._NO_THINKING_MODELS)
    vt._NO_THINKING_MODELS.clear()
    try:
        provider = vt.OllamaProvider({"provider": "ollama", "model": "fake:1b"})
        out = provider.generate("hi", thinking=True)
        check("the call still returns a usable reply", out == '{"ok":1}')
        check("it retried without `think`", len(calls) == 2 and "think" not in calls[1])
        check("the model is remembered", "fake:1b" in vt._NO_THINKING_MODELS)

        provider.generate("hi again", thinking=True)
        check("a later call never sends `think` at all",
              len(calls) == 3 and "think" not in calls[2],
              "so the failed request is paid once per model, not once per batch")

        calls.clear()
        vt._NO_THINKING_MODELS.clear()

        class AlwaysFails(FakeOllama):
            def generate(self, **kwargs):
                calls.append(dict(kwargs))
                raise RuntimeError("connection refused")

        vt.ollama = AlwaysFails()
        try:
            vt.OllamaProvider({"provider": "ollama", "model": "fake:1b"}).generate(
                "hi", thinking=True)
            check("an unrelated error still propagates", False)
        except RuntimeError as e:
            check("an unrelated error still propagates", "connection refused" in str(e))
        check("and is not retried as a thinking problem", len(calls) == 1)
    finally:
        vt.ollama = real_ollama
        vt._NO_THINKING_MODELS.clear()
        vt._NO_THINKING_MODELS.update(remembered)


def test_model_residency():
    section("model residency -- never unload a model the next step reloads")
    ollama_a = vt.OllamaProvider({"provider": "ollama", "model": "a:1b",
                                  "url": "http://localhost:11434"})
    ollama_a2 = vt.OllamaProvider({"provider": "ollama", "model": "a:1b",
                                   "url": "http://localhost:11434"})
    ollama_b = vt.OllamaProvider({"provider": "ollama", "model": "b:1b",
                                  "url": "http://localhost:11434"})
    remote_a = vt.OllamaProvider({"provider": "ollama", "model": "a:1b",
                                  "url": "http://box:11434"})
    check("the same model on the same server is one residency",
          vt._same_local_model(ollama_a, ollama_a2))
    check("a different model is not", not vt._same_local_model(ollama_a, ollama_b))
    check("the same name on a different server is not",
          not vt._same_local_model(ollama_a, remote_a))
    check("there is no next stage to keep it for",
          not vt._same_local_model(ollama_a, None))
    check("a cloud stage holds no local RAM",
          not vt._same_local_model(
              ollama_a, vt.OpenAIProvider({"provider": "openai", "model": "a:1b"})))

    slept = []
    real_sleep, vt.time.sleep = vt.time.sleep, lambda s: slept.append(s)
    try:
        vt._last_ollama_model = "a:1b"
        vt.wait_for_model_unload("a:1b")
        check("re-using the loaded model waits for nothing", slept == [])
    finally:
        vt.time.sleep = real_sleep
        vt._last_ollama_model = None

    calls = []

    class FakeOllama:
        def generate(self, **kwargs):
            calls.append(kwargs.get("keep_alive"))
            return {"response": '{"results": []}', "prompt_eval_count": 1,
                    "eval_count": 1}

        def ps(self):
            return {"models": []}

    real_ollama, vt.ollama = vt.ollama, FakeOllama()
    real_batch, vt.CLASSIFY_BATCH_SIZE = vt.CLASSIFY_BATCH_SIZE, 1
    events = [vt.Event(title="one"), vt.Event(title="two")]
    today, week_end = date(2026, 8, 24), date(2026, 8, 30)
    try:
        vt._last_ollama_model = None
        vt.classify(events, ollama_a, today, week_end, "worker")
        check("batches before the last keep the model in RAM",
              calls[0] == vt.KEEP_LOADED_SECONDS,
              "a cold reload per batch would cost ~6.5 min each")
        check("the last batch of the last stage unloads it", calls[-1] == 0)

        calls.clear()
        vt.classify(events, ollama_a, today, week_end, "worker", keep_after=True)
        check("the last batch stays loaded when the next stage is the same model",
              calls == [vt.KEEP_LOADED_SECONDS, vt.KEEP_LOADED_SECONDS])
    finally:
        vt.ollama = real_ollama
        vt.CLASSIFY_BATCH_SIZE = real_batch
        vt._last_ollama_model = None


def test_week_filtering():
    section("_in_week -- multi-date text is a span")
    start, end = date(2026, 8, 24), date(2026, 8, 30)
    check("a single date inside the week", vt._in_week([date(2026, 8, 26)], start, end))
    check("a single date outside it", not vt._in_week([date(2026, 9, 9)], start, end))
    check("a span overlapping the week",
          vt._in_week([date(2026, 8, 20), date(2026, 8, 26)], start, end))
    evergreen = vt.EVERGREEN_MAX_DAYS
    check("an always-on listing is dropped",
          not vt._in_week([date(2026, 1, 1), date(2026, 12, 31)], start, end),
          f"spans longer than EVERGREEN_MAX_DAYS={evergreen} days")
    check("no date at all is not in the week", not vt._in_week([], start, end))


def test_grouping():
    section("group_by_day -- possible only because dates are parsed, not restated")
    start, end = date(2026, 8, 24), date(2026, 8, 30)
    inside = vt.Event(title="Tuesday thing", dates=[date(2026, 8, 25)])
    spanning = vt.Event(title="Long run", dates=[date(2026, 8, 1), date(2026, 9, 20)])
    dateless = vt.Event(title="A notice")
    headings = [h for h, _ in vt.group_by_day([inside, spanning, dateless], start, end)]
    check("a dated event is filed under its day", "Tuesday, August 25" in headings)
    check("an event spanning the week is not backdated out of it",
          "Ongoing this week" in headings)
    check("a dateless notice is labelled as such", "Undated notices" in headings)


def test_escaping():
    section("_e -- scraped and model-written text is untrusted")
    check("tags are escaped", vt._e("<script>alert(1)</script>")
          == "&lt;script&gt;alert(1)&lt;/script&gt;")
    check("ampersands are escaped", vt._e("Cafe & Chill") == "Cafe &amp; Chill")
    check("quotes are escaped for attribute context", "&quot;" in vt._e('say "hi"'))
    check("None becomes empty, not the string 'None'", vt._e(None) == "")


def test_scrape_roundtrip():
    section("scrape cache -- Event survives a save/load cycle")
    original = vt.Event(title="T", when="w", dates=[date(2026, 8, 25)],
                        location="L", flags=["FREE FOOD"], matches=[1], reason="r")
    restored = vt._event_from_dict(vt._event_to_dict(original))
    check("every field round-trips", restored == original)
    check("dates come back as date objects, not strings",
          all(isinstance(d, date) for d in restored.dates))


def test_email_bodies():
    section("_email_bodies -- the digest")
    ev = vt.Event(title="Pizza & <b>Chess</b>", when="5pm", location="Squires",
                  link="https://x/y", flags=["FREE FOOD"], matches=[1], reason="free pizza")
    text, body = vt._email_bodies([ev], "")
    check("no report link when no report was written", "Full list" not in text,
          "email-only mode must not link to a file that does not exist")
    check("html body escapes the title", "&lt;b&gt;" in body)
    text2, _ = vt._email_bodies([ev], "http://host/r.html")
    check("the report link appears when there is one",
          "Full list: http://host/r.html" in text2)
    empty_text, _ = vt._email_bodies([], "")
    check("an empty digest still renders", "No high-interest events" in empty_text)


def test_criteria_normalization():
    section("_normalize_criteria -- both shorthand and full form")
    out = vt._normalize_criteria(["plain", {"text": "rich", "auto_match_flags": ["F"]}])
    check("a plain string becomes a numbered criterion",
          out[0] == {"n": 1, "text": "plain", "auto_match_flags": []})
    check("an object keeps its claimed flags",
          out[1] == {"n": 2, "text": "rich", "auto_match_flags": ["F"]})


# Trimmed to the shape that matters: two semester tables the parser must read,
# and the Career Services office-hours table it must not. The real page adds
# prose and chrome around these, none of which the parser looks at.
_FAIR_HTML = """
<h2>Fall 2026</h2>
<table>
  <tr><th>Name</th><th>Date</th></tr>
  <tr><td><a href="https://joinhandshake.com/f/1">Meet the Firms</a></td>
      <td>September 8</td></tr>
  <tr><td><a href="/students/bh.html">Business Horizons Career Fair</a></td>
      <td>September 9-10</td></tr>
  <tr><td>Engineering Expo</td><td>September 15-17</td></tr>
</table>
<h2>Spring 2027</h2>
<table>
  <tr><th>Name</th><th>Date</th></tr>
  <tr><td>CLAHS Career Fair</td><td>February 17</td></tr>
</table>
<h2>Career Service Hours</h2>
<table>
  <tr><th>Day of the week</th><th>Summer AM Hours</th><th>Summer PM Hours</th></tr>
  <tr><td>M Monday</td><td>8:00 am - 12:00 pm</td><td>1:00 pm - 5:00 pm</td></tr>
</table>
"""


def test_career_fair_dates():
    section("_parse_fair_dates -- the date cell carries no year, and may be a range")
    check("a single day",
          vt._parse_fair_dates("September 8", 2026) == [date(2026, 9, 8)])
    check("a same-month range keeps its end day",
          vt._parse_fair_dates("September 9-10", 2026)
          == [date(2026, 9, 9), date(2026, 9, 10)],
          "_DATE_RE alone sees only the 9, so the 10th would be lost")
    check("an en dash is a range too",
          vt._parse_fair_dates("October 5\u20136", 2026)
          == [date(2026, 10, 5), date(2026, 10, 6)])
    check("the year comes from the caller, not the cell",
          vt._parse_fair_dates("February 17", 2027) == [date(2027, 2, 17)])
    check("a cross-month range already parses as two dates",
          vt._parse_fair_dates("September 30 - October 2", 2026)
          == [date(2026, 9, 30), date(2026, 10, 2)])
    check("an impossible day is dropped rather than raising",
          vt._parse_fair_dates("February 30-31", 2026) == [])
    check("a cell with no date at all", vt._parse_fair_dates("TBD", 2026) == [])


def test_career_fair_parsing():
    section("fetch_career_fairs -- semester tables, not a uConnect feed")
    source = {"id": "career_fairs", "name": "VT Career Fairs",
              "url": "https://career.vt.edu/resources/career-fairs/"}
    real_get, vt._get = vt._get, lambda url, as_json=False: _FAIR_HTML
    try:
        fall = vt.fetch_career_fairs(source, date(2026, 9, 7), date(2026, 9, 13))
        spanning = vt.fetch_career_fairs(source, date(2026, 9, 16), date(2026, 9, 20))
        spring = vt.fetch_career_fairs(source, date(2027, 2, 15), date(2027, 2, 21))
        hours = vt.fetch_career_fairs(source, date(2026, 6, 1), date(2026, 6, 7))
    finally:
        vt._get = real_get

    titles = [e.title for e in fall]
    check("fairs inside the week are collected",
          titles == ["Meet the Firms", "Business Horizons Career Fair"],
          f"got {titles}")
    check("a fair whose span reaches into the week is not missed",
          [e.title for e in spanning] == ["Engineering Expo"],
          "September 15-17 starts before a week beginning the 16th")
    check("the year is read from the section heading, not the run's year",
          [e.dates for e in spring] == [[date(2027, 2, 17)]],
          "a Spring 2027 row must not be dated 2026")
    check("the office-hours table is not parsed as events", hours == [],
          "its header is not Name | Date")
    check("a relative href is resolved against the page",
          fall[1].link == "https://career.vt.edu/students/bh.html")
    check("an absolute href is left alone",
          fall[0].link == "https://joinhandshake.com/f/1")
    check("the source id is carried through",
          all(e.source_id == "career_fairs" for e in fall))


# Four real li.event-page items from https://events.vt.edu/events/2026.html,
# captured 2026-09-16, with the image/srcset wrappers cut out. The class
# attributes are verbatim: they are the authoritative facets the parser reads,
# so trimming those would defeat the point of the fixture. The last <li> repeats
# the Eric Lyon href to exercise the de-duplication guard.
_EVENTS_VT_HTML = """
<ul>
<li class="col-12 col-md-6 item locations_-blacksburg-va-24061 categories_-concert departments_-school-of-performing-arts types_-in-person audiences_-public event-page">
  <span class="vt-list-item-heading-text">Event Item</span>
  <span class="vt-list-item-title">
    <a href="https://events.vt.edu/events/2026/09/eric-lyon-portrait-2026.html">
      Eric Lyon Portrait Concert, featuring Annie Stevens and Alan Weinstein
      <span class="sr-only">, event</span>
    </a>
  </span>
  <span class="vt-list-item-date">
    <span class="vt-list-item-dateHeading">Date: </span>
    <span class="vt-list-item-date-start">Sep 16, 2026</span>
  </span>
</li>
<li class="col-12 col-md-6 item types_-hybrid categories_-meeting departments_-virginia-tech-carilion-school-of-medicine admission_-free audiences_-public event-page">
  <span class="vt-list-item-heading-text">Event Item</span>
  <span class="vt-list-item-title">
    <a href="https://events.vt.edu/events/2026/09/staff-senate-monthly-meeting.html">
      Staff Senate Monthly Meeting<span class="sr-only">, event</span>
    </a>
  </span>
  <span class="vt-list-item-date">
    <span class="vt-list-item-dateHeading">Date: </span>
    <span class="vt-list-item-date-start">Sep 17, 2026</span>
  </span>
</li>
<li class="col-12 col-md-6 item categories_-open-house locations_-blacksburg-va-24061 audiences_-faculty audiences_-public features_-free-food features_-free-parking features_-interactive-activities types_-in-person departments_-research-and-innovation event-page">
  <span class="vt-list-item-heading-text">Event Item</span>
  <span class="vt-list-item-title">
    <a href="https://events.vt.edu/events/2026/09/ncfl-open-house-and-user-appreciation-day.html">
      NCFL Open House and User Appreciation Day<span class="sr-only">, event</span>
    </a>
  </span>
  <span class="vt-list-item-date">
    <span class="vt-list-item-dateHeading">Date: </span>
    <span class="vt-list-item-date-start">Sep 18, 2026</span>
  </span>
</li>
<li class="col-12 col-md-6 item categories_-exhibition types_-in-person departments_-research-and-innovation event-page">
  <span class="vt-list-item-heading-text">Event Item</span>
  <span class="vt-list-item-title">
    <a href="https://events.vt.edu/events/2026/10/flip-the-fair-2026.html">
      Flip the Fair 2026<span class="sr-only">, event</span>
    </a>
  </span>
  <span class="vt-list-item-date">
    <span class="vt-list-item-dateHeading">Date: </span>
    <span class="vt-list-item-date-start">Oct 01, 2026</span>
  </span>
</li>
<li class="col-12 col-md-6 item categories_-concert event-page">
  <span class="vt-list-item-heading-text">Event Item</span>
  <span class="vt-list-item-title">
    <a href="https://events.vt.edu/events/2026/09/eric-lyon-portrait-2026.html">
      Eric Lyon Portrait Concert (repeat listing)<span class="sr-only">, event</span>
    </a>
  </span>
  <span class="vt-list-item-date">
    <span class="vt-list-item-dateHeading">Date: </span>
    <span class="vt-list-item-date-start">Sep 16, 2026</span>
  </span>
</li>
</ul>
"""

_DETAIL_HTML = "<main><p>A description from the detail page.</p></main>"


def _events_vt_fake_get(listing):
    """Serve the year listing for a listing URL, a stub for any detail page."""
    def fake(url, as_json=False):
        tail = url.rsplit("/", 1)[-1]          # "2026.html" or "<slug>.html"
        return listing if tail[:-5].isdigit() else _DETAIL_HTML
    return fake


def test_events_vt_listing_urls():
    section("_events_vt_listing_urls -- the listing is per year, not per month")
    check("one URL for a week inside a single year",
          vt._events_vt_listing_urls(date(2026, 9, 14), date(2026, 9, 20))
          == [(2026, "https://events.vt.edu/events/2026.html")])
    check("a week spanning New Year fetches both years",
          vt._events_vt_listing_urls(date(2026, 12, 28), date(2027, 1, 3))
          == [(2026, "https://events.vt.edu/events/2026.html"),
              (2027, "https://events.vt.edu/events/2027.html")],
          "the January half of the week would otherwise be invisible")


def test_events_vt_parsing():
    section("fetch_events_vt -- year listing, facets read from the CSS classes")
    source = {"id": "events_vt", "name": "Virginia Tech Events",
              "url": "https://events.vt.edu/"}
    real_get, real_delay = vt._get, vt.POLITE_DELAY
    vt._get, vt.POLITE_DELAY = _events_vt_fake_get(_EVENTS_VT_HTML), 0
    try:
        found = vt.fetch_events_vt(source, date(2026, 9, 14), date(2026, 9, 20))
    finally:
        vt._get, vt.POLITE_DELAY = real_get, real_delay

    titles = [e.title for e in found]
    check("only the in-week items are collected",
          titles == ["Eric Lyon Portrait Concert, featuring Annie Stevens "
                     "and Alan Weinstein",
                     "Staff Senate Monthly Meeting",
                     "NCFL Open House and User Appreciation Day"],
          f"got {titles}")
    check("a title containing a comma survives the ', event' split",
          titles and titles[0].endswith("Annie Stevens and Alan Weinstein"),
          "splitting on the first comma would truncate it at 'Concert'")
    check("an out-of-week item is dropped", "Flip the Fair 2026" not in titles,
          "October 1 is not in the week of September 14")
    check("a repeated href is listed once",
          sum(1 for t in titles if t.startswith("Eric Lyon")) == 1)

    by_title = {e.title.split(",")[0]: e for e in found}
    ncfl = by_title.get("NCFL Open House and User Appreciation Day")
    check("free food is read from the listing class, not from a description",
          ncfl and ncfl.flags == ["FREE FOOD", "free parking"],
          f"got {ncfl.flags if ncfl else None}")
    check("the location facet becomes the location",
          ncfl and ncfl.location == "blacksburg va 24061",
          ncfl.location if ncfl else "")
    check("the category facet becomes the category",
          ncfl and ncfl.category == "open house", ncfl.category if ncfl else "")
    check("the department facet becomes the host",
          ncfl and ncfl.host == "research and innovation",
          ncfl.host if ncfl else "")
    check("admission_-free is a flag too",
          by_title["Staff Senate Monthly Meeting"].flags == ["free admission"])
    check("an item with no feature classes carries no flags",
          by_title["Eric Lyon Portrait Concert"].flags == [])
    check("the date is parsed, so the report can group by day",
          ncfl and ncfl.dates == [date(2026, 9, 18)],
          str(ncfl.dates) if ncfl else "")
    check("the event links to its own page, not to the listing",
          ncfl and ncfl.link.endswith("/ncfl-open-house-and-user-appreciation-day.html"))
    check("the detail page supplies the description",
          ncfl and "description from the detail page" in ncfl.description,
          ncfl.description[:60] if ncfl else "")


def test_events_vt_selector_miss():
    section("fetch_events_vt -- a 200 that matches nothing is not a quiet week")
    import io, contextlib
    source = {"id": "events_vt", "name": "Virginia Tech Events",
              "url": "https://events.vt.edu/"}
    real_get = vt._get
    vt._get = lambda url, as_json=False: "<html><body><p>redesigned</p></body></html>"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            found = vt.fetch_events_vt(source, date(2026, 9, 14), date(2026, 9, 20))
    finally:
        vt._get = real_get
    out = buf.getvalue()
    check("nothing is collected", found == [])
    check("the miss is reported as a WARNING", "WARNING" in out, out)
    check("the message names the selector that stopped matching",
          vt.EVENTS_VT_ITEM_SEL in out, out)
    check("the message names the URL", "events/2026.html" in out, out)


def test_config_json_errors():
    section("_read_json -- a hand-edited config typo is an expected failure")
    import tempfile, os
    bad = os.path.join(tempfile.mkdtemp(), "sources.json")
    with open(bad, "w", encoding="utf-8") as fh:
        fh.write('{\n  "sources": [\n    {\n      "url": "http://a"\n'
                 '      "type": "news"\n    }\n  ]\n}\n')
    try:
        vt._read_json(bad, "The sources file")
        check("a malformed file exits", False, "it returned instead")
    except SystemExit as e:
        msg = str(e)
        check("a malformed file exits rather than raising JSONDecodeError", True)
        check("the message names the file", bad in msg, msg)
        check("the message names the line", "line 5" in msg, msg)

    try:
        vt._require_key({"srcs": []}, "sources", bad, "The sources file")
        check("a missing top-level key exits", False, "it returned instead")
    except SystemExit as e:
        check("a missing top-level key exits", True)
        check("the message names the key", '"sources" key' in str(e), str(e))


if __name__ == "__main__":
    for test in (test_parse_classification, test_dedupe, test_auto_matches,
                 test_permanent_errors, test_thinking_fallback,
                 test_model_residency,
                 test_week_filtering, test_grouping,
                 test_escaping, test_scrape_roundtrip, test_email_bodies,
                 test_criteria_normalization,
                 test_career_fair_dates, test_career_fair_parsing,
                 test_events_vt_listing_urls, test_events_vt_parsing,
                 test_events_vt_selector_miss,
                 test_config_json_errors):
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")
