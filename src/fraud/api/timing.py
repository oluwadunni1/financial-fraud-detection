"""Per-request timing, carried in a standard `Server-Timing` header.

A single latency number cannot tell the model apart from the network. From a
Studio in us-east-1 every database call to Supabase in eu-west-1 costs ~70 ms
before Postgres does any work, so a total alone would describe the Atlantic
rather than the design. The header splits a request into spans, each with the
number of database round trips it made, so the replay can subtract measured
network time and project what a same-region deploy would see.

The header is the W3C format (`name;dur=1.23`), so a browser's network panel
reads it too, and the response body stays exactly as it was.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Spans:
    """Named durations for one request, in ms, plus each span's round trips."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self.round_trips: dict[str, int] = {}

    @contextmanager
    def span(self, name: str, round_trips: int = 0) -> Iterator[None]:
        """Time a block. `round_trips` is how many database calls it makes --
        declared by the code making them, so the count is never guessed."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (
                time.perf_counter() - started
            ) * 1000.0
            self.round_trips[name] = self.round_trips.get(name, 0) + round_trips

    def header(self) -> str:
        return format_server_timing(self.ms, self.round_trips)


def format_server_timing(
    spans: dict[str, float], round_trips: dict[str, int] | None = None
) -> str:
    """`fetch;dur=140.123;rt=2, score;dur=4.321;rt=0`.

    `rt` is a custom parameter. The spec allows extra parameters and browsers
    ignore what they do not know, so the header stays valid.
    """
    round_trips = round_trips or {}
    return ", ".join(
        f"{name};dur={ms:.3f};rt={round_trips.get(name, 0)}"
        for name, ms in spans.items()
    )


def parse_server_timing(header: str) -> dict[str, dict[str, float]]:
    """`fetch;dur=1.2;rt=2` -> {"fetch": {"dur": 1.2, "rt": 2.0}}."""
    spans: dict[str, dict[str, float]] = {}
    for part in filter(None, (p.strip() for p in header.split(","))):
        name, *params = (x.strip() for x in part.split(";"))
        values = {"dur": 0.0, "rt": 0.0}
        for param in params:
            key, _, value = param.partition("=")
            if key.strip() in values:
                values[key.strip()] = float(value)
        spans[name] = values
    return spans
