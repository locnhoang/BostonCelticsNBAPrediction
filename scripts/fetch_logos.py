"""One-time download of 128px team logos into site/logos/{ABBR}.png.

NOTE (2026-11-15): the committed logos were replaced by user-supplied artwork from
nba_logos_2026-27/ (resized to 128px). Re-running this script would overwrite them.

Team logos are trademarks of the NBA and its teams; they are shown on the site
under the credit in the footer. Run: uv run python scripts/fetch_logos.py
"""

import sys
from pathlib import Path

import requests

from celtics.teams import TEAMS

URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nba/500/{code}.png&h=128&w=128"
ESPN_CODE = {"GSW": "gs", "NOP": "no", "NYK": "ny", "SAS": "sa", "UTA": "utah", "WAS": "wsh"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OUT = Path(__file__).resolve().parent.parent / "site" / "logos"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for team in TEAMS:
        code = ESPN_CODE.get(team, team.lower())
        resp = requests.get(URL.format(code=code), timeout=30)
        if resp.status_code != 200 or not resp.content.startswith(PNG_SIGNATURE):
            failed.append(f"{team} ({code}): HTTP {resp.status_code}")
            continue
        (OUT / f"{team}.png").write_bytes(resp.content)
        print(f"{team}: {len(resp.content)} bytes")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
