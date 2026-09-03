"""Shared command-line configuration for the example applications."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .engine import DEFAULT_ASSET_ROOT, SupertonicModalix
from .inputs import DEFAULT_STEPS


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    compiled_dir = DEFAULT_ASSET_ROOT / "supertonic-3-sima"
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_ASSET_ROOT / "supertonic-3"
    )
    parser.add_argument(
        "--mpk",
        type=Path,
        default=compiled_dir / "supertonic_vector_field_sima_mpk.tar.gz",
    )
    parser.add_argument(
        "--runtime-data",
        type=Path,
        default=compiled_dir / "supertonic_runtime_data.npz",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=compiled_dir / "artifact_manifest.json",
    )
    parser.add_argument(
        "--vocoder-backend", choices=("mla", "cpu"), default="mla"
    )
    parser.add_argument(
        "--vocoder-mpk",
        type=Path,
        default=compiled_dir / "supertonic_vocoder_sima_bf16_mpk.tar.gz",
    )
    parser.add_argument(
        "--vocoder-manifest",
        type=Path,
        default=compiled_dir / "vocoder_bf16_manifest.json",
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--steps",
        type=int,
        choices=range(5, 13),
        default=DEFAULT_STEPS,
        help="Euler denoising steps (5-12; default: 8)",
    )
    parser.add_argument("--skip-hash-check", action="store_true")


def create_engine(args: argparse.Namespace) -> SupertonicModalix:
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    return SupertonicModalix(
        model_dir=args.model_dir,
        mpk_path=args.mpk,
        runtime_data_path=args.runtime_data,
        artifact_manifest_path=args.artifact_manifest,
        vocoder_backend=args.vocoder_backend,
        vocoder_mpk_path=args.vocoder_mpk,
        vocoder_manifest_path=args.vocoder_manifest,
        threads=args.threads,
        timeout_ms=args.timeout_ms,
        verify_hashes=not args.skip_hash_check,
        steps=args.steps,
    )
