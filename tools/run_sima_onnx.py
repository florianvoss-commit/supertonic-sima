#!/usr/bin/env python3
"""Standalone Supertonic 3 inference using the surgically rewritten ONNX.

Runtime dependencies are NumPy and ONNX Runtime only.  No import from the
``supertonic`` Python package is made.  This runner intentionally accepts one
static-profile utterance at a time; callers can chunk longer text externally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import wave
from unicodedata import normalize
from typing import Any

import numpy as np
import onnxruntime as ort

from sinusoidal_inputs import (
    CONDITIONAL_STYLE_KEY,
    MAX_SEQUENCE_LENGTH,
    ROPE_INPUT,
    ROPE_TABLE_KEY,
    RUNTIME_CONSTANT_KEYS,
    TIME_INPUT,
    TIME_TABLE_KEY,
    UNCONDITIONAL_STYLE_KEY,
    UNCONDITIONAL_STYLE_VALUE,
    UNCONDITIONAL_TEXT,
    pack_rope_input,
    validate_rope_bank,
)


AVAILABLE_LANGUAGES = {
    "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hi", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "na", "nl",
    "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
}
MIN_SPEED = 0.7
MAX_SPEED = 2.0

_EMOJI_PATTERN = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff\u2600-\u26ff\u2700-\u27bf"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)
_SYMBOL_REPLACEMENTS = {
    "\u2013": "-", "\u2011": "-", "\u2014": "-", "\u00af": " ", "_": " ",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u00b4": "'",
    "`": "'", "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
    "→": " ", "←": " ",
}
_SPECIAL_SYMBOLS = re.compile(r"[♥☆♡©\\]")
_DUPLICATE_QUOTES = re.compile(r'(["\'`])\1+')
_WHITESPACE = re.compile(r"\s+")
_ENDING_PUNCTUATION = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    delta = candidate - reference
    reference64 = reference.reshape(-1).astype(np.float64)
    candidate64 = candidate.reshape(-1).astype(np.float64)
    delta64 = delta.reshape(-1).astype(np.float64)
    reference_norm = float(np.linalg.norm(reference64))
    candidate_norm = float(np.linalg.norm(candidate64))
    tiny = np.finfo(np.float64).tiny
    return {
        "shape_equal": reference.shape == candidate.shape,
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta64) / max(reference_norm, tiny)),
        "cosine": float(
            np.dot(reference64, candidate64)
            / max(reference_norm * candidate_norm, tiny)
        ),
    }


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def preprocess_text(text: str, language: str) -> str:
    if language not in AVAILABLE_LANGUAGES:
        raise ValueError(f"unsupported language {language!r}")
    text = normalize("NFKD", text)
    text = _EMOJI_PATTERN.sub("", text)
    for old, new in _SYMBOL_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = _SPECIAL_SYMBOLS.sub("", text)
    text = text.replace("@", " at ")
    text = text.replace("e.g.,", "for example, ")
    text = text.replace("i.e.,", "that is, ")
    for old, new in (
        (" ,", ","), (" .", "."), (" !", "!"), (" ?", "?"),
        (" ;", ";"), (" :", ":"), (" '", "'"),
    ):
        text = text.replace(old, new)
    text = _DUPLICATE_QUOTES.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        raise ValueError("text is empty after preprocessing")
    if not _ENDING_PUNCTUATION.search(text):
        text += "."
    return f"<{language}>{text}</{language}>"


def encode_text(text: str, indexer_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    indexer = json.loads(indexer_path.read_text())
    if not isinstance(indexer, list) or not indexer:
        raise ValueError("unicode_indexer.json must contain a non-empty list")
    ids = []
    unsupported = []
    for character in text:
        codepoint = ord(character)
        index = indexer[codepoint] if codepoint < len(indexer) else -1
        if index < 0:
            unsupported.append(character)
        ids.append(index)
    if unsupported:
        raise ValueError(f"unsupported characters after preprocessing: {sorted(set(unsupported))}")
    if len(ids) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"processed text length {len(ids)} exceeds static limit {MAX_SEQUENCE_LENGTH}"
        )
    text_ids = np.asarray([ids], dtype=np.int64)
    text_mask = np.ones((1, 1, len(ids)), dtype=np.float32)
    return text_ids, text_mask, text


def load_style(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    values = []
    for name, expected in (("style_ttl", (1, 50, 256)), ("style_dp", (1, 8, 16))):
        entry = payload[name]
        value = np.asarray(entry["data"], dtype=np.float32).reshape(entry["dims"])
        if value.shape != expected:
            raise ValueError(f"{name} shape {value.shape} != {expected}")
        values.append(value)
    return values[0], values[1]


def pad_last(value: np.ndarray, width: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    if value.shape[-1] > width:
        raise ValueError(f"input width {value.shape[-1]} exceeds {width}")
    padding = [(0, 0)] * value.ndim
    padding[-1] = (0, width - value.shape[-1])
    return np.ascontiguousarray(np.pad(value, padding), dtype=np.float32)


def style_nchw(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 1)[:, :, None, :])


def save_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(np.asarray(waveform).reshape(-1), -1.0, 1.0)
    pcm16 = np.rint(samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm16.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vector-field", type=Path, required=True)
    parser.add_argument("--runtime-data", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="M1")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference-case", type=Path)
    args = parser.parse_args()

    if not MIN_SPEED <= args.speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
    model_dir = args.model_dir.resolve()
    onnx_dir = model_dir / "onnx"
    config = json.loads((onnx_dir / "tts.json").read_text())
    sample_rate = int(config["ae"]["sample_rate"])
    base_chunk_size = int(config["ae"]["base_chunk_size"])
    compress = int(config["ttl"]["chunk_compress_factor"])
    latent_dim = int(config["ttl"]["latent_dim"]) * compress
    chunk_size = base_chunk_size * compress

    normalized = preprocess_text(args.text, args.lang)
    text_ids, text_mask, normalized = encode_text(
        normalized, onnx_dir / "unicode_indexer.json"
    )
    style_ttl, style_dp = load_style(model_dir / "voice_styles" / f"{args.voice}.json")

    duration_session = make_session(onnx_dir / "duration_predictor.onnx", args.threads)
    text_session = make_session(onnx_dir / "text_encoder.onnx", args.threads)
    vector_session = make_session(args.vector_field.resolve(), args.threads)
    vocoder_session = make_session(onnx_dir / "vocoder.onnx", args.threads)

    duration = duration_session.run(
        None,
        {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask},
    )[0].astype(np.float32) / args.speed
    text_emb = text_session.run(
        None,
        {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask},
    )[0].astype(np.float32)

    wav_lengths = (duration * sample_rate).astype(np.int64)
    latent_length = int(
        ((float(duration.max()) * sample_rate + chunk_size - 1) / chunk_size)
    )
    valid_latent_length = int((wav_lengths[0] + chunk_size - 1) // chunk_size)
    if latent_length > MAX_SEQUENCE_LENGTH or valid_latent_length > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"predicted latent length {max(latent_length, valid_latent_length)} "
            f"exceeds static limit {MAX_SEQUENCE_LENGTH}; increase speed or shorten text"
        )
    latent_mask = np.zeros((1, 1, latent_length), dtype=np.float32)
    latent_mask[..., :valid_latent_length] = 1.0
    rng = np.random.RandomState(args.seed)
    latent = rng.randn(1, latent_dim, latent_length).astype(np.float32) * latent_mask

    with np.load(args.runtime_data) as archive:
        time_table = np.asarray(archive[TIME_TABLE_KEY], dtype=np.float32)
        rope_table = validate_rope_bank(archive[ROPE_TABLE_KEY]).copy()
        constants = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in RUNTIME_CONSTANT_KEYS
        }
    if time_table.shape[0] != args.steps:
        raise ValueError(
            f"runtime data contains {time_table.shape[0]} steps, requested {args.steps}"
        )

    padded_text = pad_last(text_emb)
    padded_text_mask = pad_last(text_mask)
    padded_mask = pad_last(latent_mask)
    padded_latent = pad_last(latent)
    rope_input = pack_rope_input(rope_table, padded_mask, padded_text_mask)
    shared_inputs = {
        "latent_mask": padded_mask[:, :, None, :],
        "text_mask": padded_text_mask.transpose(0, 2, 1)[:, :, :, None],
        ROPE_INPUT: rope_input,
    }

    for step in range(args.steps):
        branch_outputs = []
        for conditional in (True, False):
            if conditional:
                branch_text = padded_text
                style_key = constants[CONDITIONAL_STYLE_KEY]
                style_value = style_ttl
            else:
                branch_text = np.broadcast_to(
                    constants[UNCONDITIONAL_TEXT], padded_text.shape
                ).copy()
                style_key = constants[UNCONDITIONAL_STYLE_KEY]
                style_value = constants[UNCONDITIONAL_STYLE_VALUE]
            inputs = {
                "noisy_latent": padded_latent[:, :, None, :],
                "text_emb": branch_text[:, :, None, :],
                "style_ttl": style_nchw(style_value),
                "style_key": style_nchw(style_key),
                **shared_inputs,
                TIME_INPUT: time_table[step],
            }
            branch_outputs.append(vector_session.run(None, inputs)[0][:, :, 0, :])
        conditional, unconditional = branch_outputs
        padded_latent = (
            (
                padded_latent
                + (4.0 * conditional - 3.0 * unconditional) / args.steps
            )
            * padded_mask
        ).astype(np.float32)

    final_latent = np.ascontiguousarray(padded_latent[..., :latent_length])
    waveform = vocoder_session.run(None, {"latent": final_latent})[0].astype(np.float32)
    save_wav(args.output, waveform, sample_rate)
    if args.output_npz:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_npz,
            waveform_float32=waveform,
            final_latent=final_latent,
            duration=duration,
            text_ids=text_ids,
            text_mask=text_mask,
            latent_mask=latent_mask,
        )

    report: dict[str, Any] = {
        "status": "passed",
        "text": args.text,
        "normalized_text": normalized,
        "language": args.lang,
        "voice": args.voice,
        "seed": args.seed,
        "steps": args.steps,
        "speed": args.speed,
        "text_length": int(text_ids.shape[-1]),
        "latent_length": latent_length,
        "valid_latent_length": valid_latent_length,
        "duration_seconds": float(duration[0]),
        "waveform_samples": int(waveform.shape[-1]),
        "artifacts": {
            "vector_field": {
                "path": str(args.vector_field.resolve()),
                "sha256": sha256(args.vector_field),
            },
            "runtime_data": {
                "path": str(args.runtime_data.resolve()),
                "sha256": sha256(args.runtime_data),
            },
            "wav": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
        },
    }
    if args.reference_case:
        with np.load(args.reference_case) as reference:
            report["reference_comparison"] = {
                "waveform": metrics(reference["waveform_float32"], waveform),
                "final_latent": metrics(
                    reference["final_latent_natural"], final_latent
                ),
            }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
