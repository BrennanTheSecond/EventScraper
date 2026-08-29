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
    check("a model rejecting `think`",
          vt._is_permanent(Exception('"granite4.1:8b" does not support thinking')))
    check("a bad API key", vt._is_permanent(Exception("invalid_api_key")))
    check("an unknown model", vt._is_permanent(Exception("model foo not found")))
    check("a rate limit IS retried", not vt._is_permanent(Exception("429 rate limit")))
    check("a timeout IS retried", not vt._is_permanent(Exception("Read timed out")))


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


if __name__ == "__main__":
    for test in (test_parse_classification, test_dedupe, test_auto_matches,
                 test_permanent_errors, test_week_filtering, test_grouping,
                 test_escaping, test_scrape_roundtrip, test_email_bodies,
                 test_criteria_normalization):
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")
