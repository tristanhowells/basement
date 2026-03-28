"""
racenet.com.au Horse Racing Results Scraper
===========================================
Scrapes historical Australian horse racing results from racenet.com.au,
including full "Form" tab data for each runner — intended for DNN/RL training.

Usage:
    python scrape_racenet.py [--days N] [--output DIR] [--start-date YYYY-MM-DD]

Requirements:
    pip install playwright beautifulsoup4
    playwright install chromium

Output:
    data/YYYY-MM-DD/{venue}/race_{n}.json   — one JSON file per race
    data/index.json                          — master index of all scraped races

JSON schema per race file:
    {
      "meta": { race-level fields },
      "runners": [ { per-horse fields including form history } ]
    }
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://www.racenet.com.au"
RESULTS_URL = f"{BASE_URL}/results/horse-racing"

# Australian state/territory identifiers used to filter non-AU meetings.
# Racenet URLs for AU meetings contain these track names or the state codes in
# the meeting slug.  NZ / international meetings are excluded by checking that
# the meeting URL does NOT match a known overseas pattern.
OVERSEAS_KEYWORDS = [
    "hong-kong", "singapore", "newmarket", "ascot-uk", "cheltenham",
    "goodwood", "ireland", "france", "usa", "japan", "south-africa",
    "uae", "dubai",
]

# Playwright settings
HEADLESS = True
SLOW_MO_MS = 0            # increase to ~100 if you get bot-detection issues
PAGE_TIMEOUT = 30_000      # ms
NAV_TIMEOUT  = 30_000      # ms

# Polite crawl delay between race pages (seconds)
RACE_DELAY = 1.5
MEETING_DELAY = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def is_australian(meeting_url: str) -> bool:
    """Return True if the meeting URL looks like an Australian venue."""
    lower = meeting_url.lower()
    for kw in OVERSEAS_KEYWORDS:
        if kw in lower:
            return False
    return True


def clean_text(t: Optional[str]) -> str:
    if t is None:
        return ""
    return " ".join(t.split())


def parse_float(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[^0-9.\-]", "", s))
    except (ValueError, TypeError):
        return None


def parse_int(s: str) -> Optional[int]:
    try:
        return int(re.sub(r"[^0-9\-]", "", s))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

async def wait_and_get_html(page: Page, selector: str, timeout: int = PAGE_TIMEOUT) -> str:
    try:
        await page.wait_for_selector(selector, timeout=timeout)
    except Exception:
        pass
    return await page.content()


async def click_tab(page: Page, tab_text: str) -> bool:
    """Click a tab button whose visible text matches tab_text (case-insensitive)."""
    try:
        # Common patterns racenet uses for tab buttons
        selectors = [
            f"button:has-text('{tab_text}')",
            f"[role='tab']:has-text('{tab_text}')",
            f"a:has-text('{tab_text}')",
            f"li:has-text('{tab_text}')",
        ]
        for sel in selectors:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                await btn.click()
                await page.wait_for_load_state("networkidle", timeout=10_000)
                return True
    except Exception as exc:
        log.debug("Tab click failed for '%s': %s", tab_text, exc)
    return False


# ---------------------------------------------------------------------------
# Page-level scrapers
# ---------------------------------------------------------------------------

async def get_meeting_urls_for_date(page: Page, target_date: date) -> list[dict]:
    """
    Navigate to the results listing for *target_date* and return a list of
    dicts: {"venue": str, "url": str, "date": str}
    """
    date_str = slugify_date(target_date)
    url = f"{RESULTS_URL}?date={target_date.isoformat()}"
    log.info("Loading results index for %s → %s", target_date, url)

    await page.goto(url, timeout=NAV_TIMEOUT)
    await page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)

    # Give React a moment to hydrate
    await asyncio.sleep(1.5)

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    meetings = []

    # Racenet renders meeting cards as <a href="/results/horse-racing/{slug}">
    # We collect every unique href that matches the results path + a date slug.
    pattern = re.compile(
        r"/results/horse-racing/([^/]+)-(" + date_str + r")/?$"
    )

    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = pattern.search(href)
        if m and href not in seen:
            seen.add(href)
            venue_slug = m.group(1)
            full_url = BASE_URL + href if href.startswith("/") else href
            if is_australian(full_url):
                venue_name = venue_slug.replace("-", " ").title()
                meetings.append({
                    "venue": venue_name,
                    "venue_slug": venue_slug,
                    "url": full_url,
                    "date": target_date.isoformat(),
                })

    if not meetings:
        log.warning("No Australian meetings found for %s (check selectors)", target_date)

    log.info("Found %d Australian meetings for %s", len(meetings), target_date)
    return meetings


async def get_race_urls_for_meeting(page: Page, meeting: dict) -> list[dict]:
    """
    Load the meeting page and collect URLs for every individual race.
    Returns a list of dicts: {"race_number": int, "race_name": str, "url": str}
    """
    log.info("Loading meeting: %s", meeting["url"])
    await page.goto(meeting["url"], timeout=NAV_TIMEOUT)
    await page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
    await asyncio.sleep(1.0)

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    races = []
    seen = set()

    # Pattern: /results/horse-racing/{venue}-{date}/{race-slug}-race-{n}
    pattern = re.compile(
        r"/results/horse-racing/[^/]+-\d{8}/(.+-race-(\d+))/?$"
    )

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = pattern.search(href)
        if m and href not in seen:
            seen.add(href)
            race_slug = m.group(1)
            race_number = int(m.group(2))
            full_url = BASE_URL + href if href.startswith("/") else href
            races.append({
                "race_number": race_number,
                "race_slug": race_slug,
                "url": full_url,
            })

    races.sort(key=lambda r: r["race_number"])
    log.info("  Found %d races at %s", len(races), meeting["venue"])
    return races


async def scrape_race_results_tab(soup: BeautifulSoup) -> dict:
    """Extract race metadata and the finishing order from the Results tab."""
    meta = {}

    # --- Race header info ---
    # Racenet typically has something like:
    #   "Race 8 | MNP Security Benchmark 78 Handicap | 1200m | Good 4 | $45,000"
    header_selectors = [
        "[class*='raceHeader']",
        "[class*='race-header']",
        "[class*='RaceHeader']",
        "h1", "h2",
    ]
    for sel in header_selectors:
        el = soup.select_one(sel)
        if el:
            header_text = clean_text(el.get_text())
            meta["race_header_raw"] = header_text
            break

    # Distance
    dist_m = re.search(r"(\d{3,5})\s*m\b", meta.get("race_header_raw", ""), re.I)
    if dist_m:
        meta["distance_m"] = int(dist_m.group(1))

    # Track condition
    cond_m = re.search(
        r"\b(Firm|Good|Soft|Heavy|Synthetic|Slow)\s*(\d)?",
        meta.get("race_header_raw", ""), re.I
    )
    if cond_m:
        meta["track_condition"] = cond_m.group(0).strip()

    # Prize money
    prize_m = re.search(r"\$\s*([\d,]+)", meta.get("race_header_raw", ""))
    if prize_m:
        meta["prize_money"] = int(prize_m.group(1).replace(",", ""))

    # --- Runners table ---
    runners = []
    table = soup.find("table")
    if table:
        headers = [clean_text(th.get_text()).lower() for th in table.find_all("th")]
        for tr in table.find_all("tr")[1:]:
            cells = [clean_text(td.get_text()) for td in tr.find_all("td")]
            if not cells:
                continue
            row = dict(zip(headers, cells)) if headers else {}
            # Fallback positional parse if headers missing
            runner = {
                "finish_position": row.get("pos") or row.get("position") or (cells[0] if cells else None),
                "barrier": row.get("barrier") or row.get("draw") or None,
                "horse_name": row.get("horse") or row.get("name") or (cells[2] if len(cells) > 2 else None),
                "jockey": row.get("jockey") or row.get("rider") or None,
                "trainer": row.get("trainer") or None,
                "weight": row.get("weight") or row.get("wt") or None,
                "margin": row.get("margin") or row.get("margins") or None,
                "win_odds": row.get("odds") or row.get("win") or row.get("sp") or None,
            }
            # Clean up
            for k, v in runner.items():
                if isinstance(v, str):
                    runner[k] = clean_text(v) or None
            runners.append(runner)

    return {"meta": meta, "runners": runners}


async def scrape_form_tab(soup: BeautifulSoup, runners_so_far: list[dict]) -> list[dict]:
    """
    Extract per-horse form history from the Form tab HTML.
    Merges the form data into the existing runners list by horse name.
    """
    form_by_horse: dict[str, list] = {}

    # Racenet renders form as expandable rows or a separate section per horse.
    # Common structure: a div/section per horse with a nested table of past runs.
    # We look for any element whose text contains a horse name we've seen,
    # then find a sibling/child table.

    known_names = {
        r["horse_name"].upper(): r["horse_name"]
        for r in runners_so_far
        if r.get("horse_name")
    }

    # Strategy: find all tables that appear after a heading containing a horse name
    # or within a card/section labelled with a horse name.
    sections = soup.find_all(
        lambda tag: tag.name in ("div", "section", "article")
        and any(
            name in tag.get_text().upper()
            for name in known_names
        )
    )

    # Also handle simple case: one big form table with horse names as sub-headers
    # Parse every <table> and try to associate it with a horse
    all_tables = soup.find_all("table")

    for table in all_tables:
        # Look for a preceding heading that matches a horse name
        horse_name = None
        prev = table.find_previous(["h3", "h4", "h5", "strong", "b", "span"])
        if prev:
            candidate = clean_text(prev.get_text()).upper()
            for known_upper, known_orig in known_names.items():
                if known_upper in candidate or candidate in known_upper:
                    horse_name = known_orig
                    break

        if not horse_name:
            continue

        headers = [clean_text(th.get_text()).lower() for th in table.find_all("th")]
        past_runs = []
        for tr in table.find_all("tr")[1:]:
            cells = [clean_text(td.get_text()) for td in tr.find_all("td")]
            if not cells:
                continue
            run = dict(zip(headers, cells)) if headers else {}
            # Normalise common column names
            past_runs.append({
                "date": run.get("date") or run.get("race date") or (cells[0] if cells else None),
                "venue": run.get("track") or run.get("venue") or run.get("course") or None,
                "distance_m": parse_int(run.get("dist") or run.get("distance") or ""),
                "track_condition": run.get("cond") or run.get("condition") or run.get("going") or None,
                "race_class": run.get("class") or run.get("race class") or None,
                "barrier": parse_int(run.get("barrier") or run.get("bar") or ""),
                "weight": run.get("weight") or run.get("wt") or None,
                "jockey": run.get("jockey") or run.get("rider") or None,
                "finish_position": run.get("pos") or run.get("position") or None,
                "margin": run.get("margin") or None,
                "time": run.get("time") or run.get("sectional") or None,
                "prize_money": parse_int(
                    re.sub(r"[^0-9]", "", run.get("prize", "") or run.get("prizemoney", "") or "")
                ),
                "win_odds": parse_float(run.get("odds") or run.get("sp") or ""),
                "rating": run.get("rating") or run.get("rtg") or None,
            })

        if past_runs:
            form_by_horse[horse_name] = past_runs

    # Merge form into runners
    enriched = []
    for runner in runners_so_far:
        name = runner.get("horse_name")
        runner["form_history"] = form_by_horse.get(name, [])
        enriched.append(runner)

    return enriched


async def scrape_race_page(page: Page, race: dict, meeting: dict) -> Optional[dict]:
    """
    Full scrape of a single race page:
      1. Load the race URL (lands on Results tab by default)
      2. Scrape results tab
      3. Click Form tab and scrape form data
      4. Return structured dict
    """
    url = race["url"]
    log.info("    Scraping race %d: %s", race["race_number"], url)

    try:
        await page.goto(url, timeout=NAV_TIMEOUT)
        await page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
        await asyncio.sleep(1.0)
    except Exception as exc:
        log.error("Failed to load race page %s: %s", url, exc)
        return None

    # ---- Results tab (default) ----
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    results_data = await scrape_race_results_tab(soup)

    meta = results_data["meta"]
    meta.update({
        "url": url,
        "race_number": race["race_number"],
        "race_slug": race.get("race_slug", ""),
        "venue": meeting["venue"],
        "venue_slug": meeting["venue_slug"],
        "date": meeting["date"],
        "scraped_at": datetime.utcnow().isoformat() + "Z",
    })

    runners = results_data["runners"]

    # ---- Form tab ----
    form_clicked = await click_tab(page, "Form")
    if form_clicked:
        await asyncio.sleep(1.5)
        form_html = await page.content()
        form_soup = BeautifulSoup(form_html, "html.parser")
        runners = await scrape_form_tab(form_soup, runners)
    else:
        log.debug("    Could not click Form tab for %s", url)
        for r in runners:
            r["form_history"] = []

    return {
        "meta": meta,
        "runners": runners,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_race(race_data: dict, output_dir: Path) -> Path:
    date_str = race_data["meta"]["date"]
    venue_slug = race_data["meta"]["venue_slug"]
    race_num = race_data["meta"]["race_number"]

    out_dir = output_dir / date_str / venue_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"race_{race_num:02d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(race_data, f, indent=2, ensure_ascii=False)

    return out_path


def update_index(index_path: Path, race_data: dict) -> None:
    index: list = []
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

    entry = {
        "date": race_data["meta"]["date"],
        "venue": race_data["meta"]["venue"],
        "venue_slug": race_data["meta"]["venue_slug"],
        "race_number": race_data["meta"]["race_number"],
        "race_slug": race_data["meta"].get("race_slug", ""),
        "url": race_data["meta"]["url"],
        "runners_count": len(race_data["runners"]),
    }

    # Replace existing entry if re-running
    index = [e for e in index if not (
        e["date"] == entry["date"]
        and e["venue_slug"] == entry["venue_slug"]
        and e["race_number"] == entry["race_number"]
    )]
    index.append(entry)
    index.sort(key=lambda e: (e["date"], e["venue_slug"], e["race_number"]))

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

async def crawl(
    days: int,
    output_dir: Path,
    start_date: Optional[date] = None,
    resume: bool = True,
) -> None:
    """
    Crawl `days` days of results, going backwards from start_date (default: yesterday).
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=1)

    dates = [start_date - timedelta(days=i) for i in range(days)]
    index_path = output_dir / "index.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index for resume logic
    scraped_keys: set = set()
    if resume and index_path.exists():
        with open(index_path, "r") as f:
            existing = json.load(f)
        scraped_keys = {
            (e["date"], e["venue_slug"], e["race_number"]) for e in existing
        }
        log.info("Resume mode: %d races already in index", len(scraped_keys))

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(
            headless=HEADLESS,
            slow_mo=SLOW_MO_MS,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context: BrowserContext = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page: Page = await context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT)

        total_races = 0

        for target_date in dates:
            log.info("=== Processing date: %s ===", target_date)

            try:
                meetings = await get_meeting_urls_for_date(page, target_date)
            except Exception as exc:
                log.error("Failed to get meetings for %s: %s", target_date, exc)
                continue

            for meeting in meetings:
                await asyncio.sleep(MEETING_DELAY)

                try:
                    races = await get_race_urls_for_meeting(page, meeting)
                except Exception as exc:
                    log.error("Failed to get races for %s: %s", meeting["url"], exc)
                    continue

                for race in races:
                    key = (meeting["date"], meeting["venue_slug"], race["race_number"])
                    if resume and key in scraped_keys:
                        log.debug("  Skipping already-scraped race %s", key)
                        continue

                    await asyncio.sleep(RACE_DELAY)

                    try:
                        race_data = await scrape_race_page(page, race, meeting)
                    except Exception as exc:
                        log.error("Error scraping race %s: %s", race["url"], exc)
                        continue

                    if race_data:
                        out_path = save_race(race_data, output_dir)
                        update_index(index_path, race_data)
                        scraped_keys.add(key)
                        total_races += 1
                        log.info("    Saved → %s", out_path)

        await browser.close()

    log.info("Done. Scraped %d races total.", total_races)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


async def _run(days=7, output="data", start_date=None, resume=True, visible=False):
    """Awaitable entry point — works in both Jupyter (await) and CLI (asyncio.run)."""
    global HEADLESS
    if visible:
        HEADLESS = False
    sd = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    await crawl(days=days, output_dir=Path(output), start_date=sd, resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape racenet.com.au horse racing results for DNN/RL training data."
    )
    parser.add_argument("--days", type=int, default=7,
                        help="Number of past days to scrape (default: 7)")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date as YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-scrape even if already in index")
    parser.add_argument("--visible", action="store_true",
                        help="Run browser in visible (non-headless) mode for debugging")
    args = parser.parse_args()

    asyncio.run(_run(
        days=args.days,
        output=args.output,
        start_date=args.start_date,
        resume=not args.no_resume,
        visible=args.visible,
    ))


if __name__ == "__main__":
    if _is_jupyter():
        # ----------------------------------------------------------------
        # Jupyter usage — edit these defaults then run the cell:
        #
        #   await _run(days=7, output="data")
        #
        # ----------------------------------------------------------------
        print("Jupyter detected. Run the scraper with:\n\n  await _run(days=7, output='data')\n")
    else:
        main()
