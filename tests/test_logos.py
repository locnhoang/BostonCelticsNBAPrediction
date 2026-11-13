from pathlib import Path

import pytest

from celtics.teams import TEAMS

LOGOS = Path(__file__).resolve().parent.parent / "site" / "logos"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("team", TEAMS)
def test_logo_is_small_valid_png(team):
    path = LOGOS / f"{team}.png"
    assert path.exists(), f"missing {path}"
    data = path.read_bytes()
    assert data.startswith(PNG_SIGNATURE)
    assert len(data) < 64_000
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert width <= 128 and height <= 128
