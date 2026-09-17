"""career.vt.edu events from its published RSS feed instead of scraped markup.

An example of `type: "module"` -- a source whose parser needs real logic rather
than a selector map. Two things make it one.

The feed is a published *contract*: career.vt.edu/events/feed/ is WordPress /
uConnect RSS, so titles, links and descriptions arrive structured and cannot be
lost to a CSS class rename the way `li.event_item` can. But its <pubDate> is the
date the listing was *published*, not the date of the event -- on 2026-09-16 all
ten items carried the same pubDate, a few minutes apart. The event date is in
the permalink instead (/events/2026/11/03/<slug>/), and reading a date out of a
URL is not something a selector map can express.

Parsed with xml.etree rather than BeautifulSoup: BeautifulSoup's XML mode needs
lxml, which this project does not depend on, and its html.parser treats <link>
as a void element -- so every item's link would come back empty.

Verified 2026-09-16: 10 items, and the permalink date matches the date the
detail page displays ("November 3, 2026" for .../2026/11/03/...).

Enable it by adding to sources.json:

    { "id": "career_rss", "name": "VT Career Center (RSS)",
      "url": "https://career.vt.edu/events/feed/",
      "type": "module", "module": "collectors/career_rss.py" }
"""

import re
import xml.etree.ElementTree as ET
from datetime import date

# /events/2026/11/03/campus-internexp-student-information-session/
_PERMALINK_DATE = re.compile(r"/events/(\d{4})/(\d{2})/(\d{2})/")
_TAGS = re.compile(r"<[^>]+>")


def _event_date(link):
    """The event's date, read from its permalink -- pubDate is the wrong date."""
    found = _PERMALINK_DATE.search(link or "")
    if not found:
        return []
    try:
        return [date(*(int(part) for part in found.groups()))]
    except ValueError:
        return []          # /2026/13/40/ parses as three integers but no date


def fetch(source, week_start, week_end):
    """Collect this week's items. `vt` is this script, injected by the loader."""
    url = source["url"]
    try:
        raw = vt._get(url)
    except Exception as e:
        print(f"  [error] {url}: {e}")
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"  [error] {url}: not valid XML: {e}")
        return []

    events, seen = [], set()
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        dates = _event_date(link)
        if not dates:
            # An item whose permalink carries no date cannot be week-filtered,
            # and a feed is not worth guessing over -- the site lists it too.
            continue
        if not vt._in_week(dates, week_start, week_end) or link in seen:
            continue
        seen.add(link)

        title = vt._clean(item.findtext("title") or "")
        if not title:
            continue
        summary = vt._clean(_TAGS.sub(" ", item.findtext("description") or ""))
        events.append(vt.Event(
            title=title[:200],
            when=dates[0].strftime("%A, %B %d, %Y"),
            dates=dates,
            description=summary[:700],
            category="Career",
            link=link,
            source_id=source["id"],
            source_url=url,
        ))
    return events
