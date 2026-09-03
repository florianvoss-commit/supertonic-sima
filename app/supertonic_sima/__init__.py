"""Hybrid Supertonic 3 runtime for SiMa Modalix."""

from .audio import save_wav, wav_bytes
from .engine import SynthesisResult, SupertonicModalix, benchmark_summary
from .text import AVAILABLE_LANGUAGES, AVAILABLE_VOICES, MAX_SPEED, MIN_SPEED

__all__ = (
    "AVAILABLE_LANGUAGES",
    "AVAILABLE_VOICES",
    "MAX_SPEED",
    "MIN_SPEED",
    "SynthesisResult",
    "SupertonicModalix",
    "benchmark_summary",
    "save_wav",
    "wav_bytes",
)
