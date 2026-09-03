#!/usr/bin/env python3
"""Generate one WAV and print latency/RTF measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supertonic_sima import benchmark_summary, save_wav  # noqa: E402
from supertonic_sima.config import add_engine_arguments, create_engine  # noqa: E402
from supertonic_sima.engine import DEFAULT_OUTPUT_ROOT  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--text", required=True)
    result.add_argument("--voice", default="M1")
    result.add_argument("--lang", default="en")
    result.add_argument("--speed", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=1101)
    result.add_argument("--warmup-runs", type=int, default=1)
    result.add_argument("--runs", type=int, default=1)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "output.wav")
    result.add_argument("--report", type=Path)
    add_engine_arguments(result)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    started = perf_counter()
    with create_engine(args) as engine:
        initialization_seconds = perf_counter() - started
        for _ in range(args.warmup_runs):
            engine.synthesize(
                args.text,
                voice=args.voice,
                language=args.lang,
                speed=args.speed,
                seed=args.seed,
            )
        results = [
            engine.synthesize(
                args.text,
                voice=args.voice,
                language=args.lang,
                speed=args.speed,
                seed=args.seed,
            )
            for _ in range(args.runs)
        ]
        result = results[-1]
        benchmark = benchmark_summary(results)
        save_wav(args.output.resolve(), result.waveform, result.sample_rate)
        report = {
            "audio_length_seconds": result.audio_seconds,
            "generation_length_seconds": result.generation_seconds,
            "real_time_factor": result.real_time_factor,
            "latent_length": result.latent_length,
            "sample_rate": result.sample_rate,
            "waveform_samples": int(result.waveform.size),
            "vocoder_backend": engine.vocoder_backend,
            "steps": engine.steps,
            "initialization_seconds": initialization_seconds,
            "warmup_runs": args.warmup_runs,
            "benchmark": benchmark,
            "stage_timings": result.timings,
            "output": str(args.output.resolve()),
        }

    print(f"audio_length_seconds={result.audio_seconds:.6f}")
    print(f"generation_length_seconds={result.generation_seconds:.6f}")
    print(f"real_time_factor={result.real_time_factor:.6f}")
    print(f"latent_length={result.latent_length}")
    for name, statistics in benchmark["stages"].items():
        print(
            f"stage={name} mean_ms={statistics['mean_ms']:.3f} "
            f"percent={statistics['mean_percent_of_generation']:.2f}"
        )
    print(f"output={args.output.resolve()}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
