from __future__ import annotations

from adaptive_3dbp.decoders.extreme_points import ExtremePointDecoder


class BottomLeftBackDecoder(ExtremePointDecoder):
    """Compatibility decoder using Extreme Points with pure z/y/x ranking."""

    name = "bottom_left_back"

