#!/usr/bin/env python3
"""Compile the saved full-BF16 vocoder AFE network with MLA tessellation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from compiler import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(default_model_name="supertonic_vocoder_sima"))
