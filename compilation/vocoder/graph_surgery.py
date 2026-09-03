#!/usr/bin/env python3
"""Create and validate the fixed length-192 all-4D vocoder graph."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from optimize_vocoder_4d import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
