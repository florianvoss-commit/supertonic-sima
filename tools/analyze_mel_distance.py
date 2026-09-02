#!/usr/bin/env python3
"""Compare WAV files with dependency-light log-mel distance metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, value = wavfile.read(path)
    if value.ndim == 2:
        value = value.mean(axis=1)
    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(value.dtype)
        scale = float(max(abs(info.min), info.max))
        value = value.astype(np.float64) / scale
    else:
        value = value.astype(np.float64)
    return int(sample_rate), value


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mel_edges = np.linspace(
        hz_to_mel(np.asarray(fmin)), hz_to_mel(np.asarray(fmax)), n_mels + 2
    )
    hz_edges = mel_to_hz(mel_edges)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = hz_edges[index : index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, np.finfo(float).tiny),
                (right - frequencies) / max(right - center, np.finfo(float).tiny),
            ),
        )
        # Area normalization prevents wide high-frequency filters dominating.
        filters[index] *= 2.0 / max(right - left, np.finfo(float).tiny)
    return filters


def mel_power(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    _, _, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    power = np.abs(spectrum) ** 2
    return mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax) @ power


def dtw_mean_cost(reference: np.ndarray, candidate: np.ndarray, radius: int) -> dict[str, float | int]:
    """Band-limited DTW over time-major log-mel frames using mean absolute dB cost."""

    n_ref, n_candidate = len(reference), len(candidate)
    radius = max(radius, abs(n_ref - n_candidate))
    accumulated = np.full((n_ref + 1, n_candidate + 1), np.inf, dtype=np.float64)
    path_length = np.zeros((n_ref + 1, n_candidate + 1), dtype=np.int32)
    accumulated[0, 0] = 0.0
    for i in range(1, n_ref + 1):
        start = max(1, i - radius)
        end = min(n_candidate, i + radius)
        for j in range(start, end + 1):
            cost = float(np.mean(np.abs(reference[i - 1] - candidate[j - 1])))
            predecessors = (
                accumulated[i - 1, j - 1],
                accumulated[i - 1, j],
                accumulated[i, j - 1],
            )
            choice = int(np.argmin(predecessors))
            if choice == 0:
                previous = (i - 1, j - 1)
            elif choice == 1:
                previous = (i - 1, j)
            else:
                previous = (i, j - 1)
            accumulated[i, j] = cost + predecessors[choice]
            path_length[i, j] = path_length[previous] + 1
    length = int(path_length[n_ref, n_candidate])
    return {
        "mean_absolute_db": float(accumulated[n_ref, n_candidate] / max(length, 1)),
        "path_frames": length,
        "radius_frames": radius,
    }


def compare_resolution(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> dict[str, object]:
    hop = n_fft // 4
    reference_mel = mel_power(reference, sample_rate, n_fft, hop, n_mels, fmin, fmax)
    candidate_mel = mel_power(candidate, sample_rate, n_fft, hop, n_mels, fmin, fmax)
    frames = min(reference_mel.shape[1], candidate_mel.shape[1])
    reference_mel = reference_mel[:, :frames]
    candidate_mel = candidate_mel[:, :frames]

    peak = max(float(reference_mel.max()), np.finfo(np.float64).tiny)
    floor = peak * 1.0e-8
    reference_db = 10.0 * np.log10(np.maximum(reference_mel, floor) / peak)
    candidate_db = 10.0 * np.log10(np.maximum(candidate_mel, floor) / peak)
    frame_energy = reference_mel.sum(axis=0)
    active = frame_energy >= max(float(frame_energy.max()) * 1.0e-5, np.finfo(float).tiny)
    if not np.any(active):
        active[:] = True
    ref_active = reference_db[:, active]
    cand_active = candidate_db[:, active]
    difference = cand_active - ref_active
    ref_shape = ref_active - ref_active.mean(axis=0, keepdims=True)
    cand_shape = cand_active - cand_active.mean(axis=0, keepdims=True)

    start = int(np.flatnonzero(active)[0])
    end = int(np.flatnonzero(active)[-1]) + 1
    dtw = dtw_mean_cost(
        reference_db[:, start:end].T,
        candidate_db[:, start:end].T,
        radius=max(4, int(round(0.20 * sample_rate / hop))),
    )
    dtw["radius_ms"] = float(1000.0 * dtw["radius_frames"] * hop / sample_rate)
    return {
        "n_fft": n_fft,
        "hop_length": hop,
        "frame_ms": 1000.0 * n_fft / sample_rate,
        "hop_ms": 1000.0 * hop / sample_rate,
        "frames": frames,
        "active_frames": int(active.sum()),
        "active_fraction": float(active.mean()),
        "log_mel_mae_db": float(np.mean(np.abs(difference))),
        "log_mel_rmse_db": float(np.sqrt(np.mean(difference**2))),
        "spectral_shape_mae_db": float(np.mean(np.abs(cand_shape - ref_shape))),
        "linear_mel_spectral_convergence": float(
            np.linalg.norm(candidate_mel[:, active] - reference_mel[:, active])
            / max(np.linalg.norm(reference_mel[:, active]), np.finfo(float).tiny)
        ),
        "dtw": dtw,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="LABEL=PATH; may be repeated",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=20000.0)
    args = parser.parse_args()

    sample_rate, reference = read_wav(args.reference)
    candidates: dict[str, Path] = {}
    for specification in args.candidate:
        label, separator, raw_path = specification.partition("=")
        if not separator or not label:
            raise ValueError(f"candidate must be LABEL=PATH: {specification}")
        candidates[label] = Path(raw_path)

    results = {}
    for label, path in candidates.items():
        candidate_rate, candidate = read_wav(path)
        if candidate_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch: {candidate_rate} != {sample_rate}")
        if candidate.shape != reference.shape:
            raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
        resolutions = [
            compare_resolution(
                reference,
                candidate,
                sample_rate,
                n_fft,
                args.n_mels,
                args.fmin,
                min(args.fmax, sample_rate / 2),
            )
            for n_fft in (1024, 2048, 4096)
        ]
        results[label] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "resolutions": resolutions,
            "multiresolution_mean": {
                key: float(np.mean([value[key] for value in resolutions]))
                for key in (
                    "log_mel_mae_db",
                    "log_mel_rmse_db",
                    "spectral_shape_mae_db",
                    "linear_mel_spectral_convergence",
                )
            },
            "dtw_log_mel_mae_db_mean": float(
                np.mean([value["dtw"]["mean_absolute_db"] for value in resolutions])
            ),
        }

    report = {
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256(args.reference),
        },
        "sample_rate": sample_rate,
        "samples": len(reference),
        "duration_seconds": len(reference) / sample_rate,
        "configuration": {
            "mel_scale": "HTK",
            "n_mels": args.n_mels,
            "fmin_hz": args.fmin,
            "fmax_hz": min(args.fmax, sample_rate / 2),
            "n_fft": [1024, 2048, 4096],
            "hop_fraction": 0.25,
            "power": 2,
            "shared_floor_db": -80.0,
            "active_threshold_db": -50.0,
        },
        "candidates": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
