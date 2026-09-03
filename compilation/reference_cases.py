#!/usr/bin/env python3
"""Generate deterministic Supertonic 3 CPU reference trajectories."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vector_estimator.contract import (
    HF_REPOSITORY,
    HF_REVISION,
    INPUT_SHAPES,
    SOURCE_SHA256,
    SUPERTONIC_VERSION,
    serializable_compiled_contract,
    serializable_contract,
)


SHORT_TEXT = "Hello from Neat GenAI Studio on the Modalix DevKit."
MEDIUM_TEXT = (
    "Neat GenAI Studio runs language, vision, speech, and retrieval workloads "
    "locally on the Modalix DevKit, keeping application data on the device."
)
SPANISH_TEXT = "Hola desde Modalix. Esta voz se genera de forma privada en el dispositivo."
NEAR_CAPACITY_SOURCE = (
    "Modalix runs speech generation locally, keeping private conversations on the device while "
    "producing a clear and responsive voice for interactive assistants, demonstrations, and "
    "multilingual applications at the edge."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_last(array: np.ndarray, target: int = 192) -> np.ndarray:
    if array.shape[-1] > target:
        raise ValueError(f"cannot pad last dimension {array.shape[-1]} to {target}")
    widths = [(0, 0)] * array.ndim
    widths[-1] = (0, target - array.shape[-1])
    return np.pad(array, widths, mode="constant").astype(np.float32, copy=False)


def calculate_latent_length(duration: np.ndarray, sample_rate: int, chunk_size: int) -> int:
    wav_len_max = float(np.max(duration)) * sample_rate
    return int(((wav_len_max + chunk_size - 1) / chunk_size))


def duration_and_lengths(core, style, text: str, language: str, speed: float) -> dict:
    text_ids, text_mask = core.text_processor([text], language)
    duration, *_ = core.dp_ort.run(
        None,
        {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask},
    )
    duration = (duration / speed).astype(np.float32)
    chunk_size = core.base_chunk_size * core.chunk_compress_factor
    latent_length = calculate_latent_length(duration, core.sample_rate, chunk_size)
    normalized = core.text_processor._preprocess_text(text, language)
    return {
        "text_ids": text_ids,
        "text_mask": text_mask.astype(np.float32),
        "duration": duration,
        "normalized_text": normalized,
        "text_length": int(text_ids.shape[-1]),
        "latent_length": latent_length,
    }


def find_near_capacity_text(core, style) -> tuple[str, dict]:
    words = NEAR_CAPACITY_SOURCE.split()
    candidates: list[tuple[int, str, dict]] = []
    for count in range(5, len(words) + 1):
        text = " ".join(words[:count]).rstrip(".,;:!?") + "."
        info = duration_and_lengths(core, style, text, "en", 1.0)
        largest = max(info["text_length"], info["latent_length"])
        if info["text_length"] <= 192 and info["latent_length"] <= 192 and largest >= 176:
            candidates.append((abs(184 - largest), text, info))
    if not candidates:
        raise RuntimeError("could not construct a near-capacity reference text")
    _, text, info = min(candidates, key=lambda item: item[0])
    return text, info


def find_exact_text_length(core, target: int) -> tuple[str, int]:
    for count in range(1, 300):
        text = "a" * count + "."
        text_ids, _ = core.text_processor([text], "en")
        if text_ids.shape[-1] == target:
            return text, count + 1
    raise RuntimeError(f"could not construct preprocessed text length {target}")


def write_pcm16_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    mono = np.asarray(waveform[0], dtype=np.float32)
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def deterministic_noisy_latent(core, duration: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        latent, mask = core.sample_noisy_latent(duration)
    finally:
        np.random.set_state(state)
    return latent.astype(np.float32), mask.astype(np.float32)


def generate_case(
    *,
    engine,
    output_dir: Path,
    name: str,
    text: str,
    language: str,
    voice: str,
    seed: int,
    speed: float = 1.0,
    total_steps: int = 8,
    expected_info: dict | None = None,
) -> dict:
    core = engine.model
    style = engine.get_voice_style(voice)
    info = expected_info or duration_and_lengths(core, style, text, language, speed)
    if info["text_length"] > 192 or info["latent_length"] > 192:
        raise ValueError(
            f"reference case {name} does not fit: T={info['text_length']} L={info['latent_length']}"
        )

    text_emb, *_ = core.text_enc_ort.run(
        None,
        {
            "text_ids": info["text_ids"],
            "style_ttl": style.ttl,
            "text_mask": info["text_mask"],
        },
    )
    text_emb = text_emb.astype(np.float32)
    noisy_latent, latent_mask = deterministic_noisy_latent(core, info["duration"], seed)

    arrays: dict[str, np.ndarray] = {
        "text_ids_natural": info["text_ids"],
        "text_mask_natural": info["text_mask"],
        "duration_seconds": info["duration"],
        "text_emb_natural": text_emb,
        "style_ttl": style.ttl.astype(np.float32),
        "style_dp": style.dp.astype(np.float32),
        "noisy_latent_natural": noisy_latent,
        "latent_mask_natural": latent_mask,
        "text_emb_padded": pad_last(text_emb),
        "text_mask_padded": pad_last(info["text_mask"]),
        "noisy_latent_padded": pad_last(noisy_latent),
        "latent_mask_padded": pad_last(latent_mask),
    }

    xt = noisy_latent.copy()
    total_step = np.array([total_steps], dtype=np.float32)
    for step in range(total_steps):
        current_step = np.array([step], dtype=np.float32)
        xt, *_ = core.vector_est_ort.run(
            None,
            {
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style.ttl,
                "text_mask": info["text_mask"],
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step,
            },
        )
        xt = xt.astype(np.float32)
        arrays[f"denoised_latent_step_{step + 1:02d}"] = xt.copy()

    waveform, *_ = core.vocoder_ort.run(None, {"latent": xt})
    waveform = waveform.astype(np.float32)
    arrays["final_latent_natural"] = xt
    arrays["waveform_float32"] = waveform

    npz_path = output_dir / f"{name}.npz"
    np.savez(npz_path, **arrays)
    wav_path = output_dir / f"{name}.wav"
    write_pcm16_wav(wav_path, waveform, core.sample_rate)

    record = {
        "name": name,
        "text": text,
        "normalized_text": info["normalized_text"],
        "raw_characters": len(text),
        "language": language,
        "voice": voice,
        "speed": speed,
        "seed": seed,
        "total_steps": total_steps,
        "checkpoints": [1, 2, 8],
        "text_length": info["text_length"],
        "latent_length": info["latent_length"],
        "predicted_duration_seconds": float(info["duration"][0]),
        "waveform_samples": int(waveform.shape[-1]),
        "waveform_duration_seconds": float(waveform.shape[-1] / core.sample_rate),
        "npz": npz_path.name,
        "npz_sha256": sha256(npz_path),
        "wav": wav_path.name,
        "wav_sha256": sha256(wav_path),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
    }
    (output_dir / f"{name}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def component_manifest(model_dir: Path) -> list[dict]:
    paths = [
        model_dir / "onnx" / "tts.json",
        model_dir / "onnx" / "unicode_indexer.json",
        model_dir / "onnx" / "duration_predictor.onnx",
        model_dir / "onnx" / "text_encoder.onnx",
        model_dir / "onnx" / "vector_estimator.onnx",
        model_dir / "onnx" / "vocoder.onnx",
        model_dir / "voice_styles" / "M1.json",
        model_dir / "voice_styles" / "F1.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model components: {missing}")
    return [
        {"path": str(path.relative_to(model_dir)), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete case JSON/NPZ/WAV artifacts after verifying their hashes",
    )
    args = parser.parse_args()

    from supertonic import TTS

    installed_version = importlib.metadata.version("supertonic")
    if installed_version != SUPERTONIC_VERSION:
        raise RuntimeError(
            f"supertonic version mismatch: expected {SUPERTONIC_VERSION}, got {installed_version}"
        )
    model_dir = args.model_dir.resolve()
    vector_hash = sha256(model_dir / "onnx" / "vector_estimator.onnx")
    if vector_hash != SOURCE_SHA256:
        raise RuntimeError(f"vector estimator hash mismatch: {vector_hash}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = TTS(
        model="supertonic-3",
        model_dir=model_dir,
        auto_download=False,
        intra_op_num_threads=8,
        inter_op_num_threads=1,
    )
    core = engine.model
    m1 = engine.get_voice_style("M1")
    near_text, near_info = find_near_capacity_text(core, m1)

    specifications = [
        ("en_m1_short", SHORT_TEXT, "en", "M1", 1101, None),
        ("en_m1_medium", MEDIUM_TEXT, "en", "M1", 1201, None),
        ("en_f1_short", SHORT_TEXT, "en", "F1", 1301, None),
        ("es_m1_short", SPANISH_TEXT, "es", "M1", 1401, None),
        ("en_m1_near_capacity", near_text, "en", "M1", 1501, near_info),
    ]
    cases = []
    for name, text, language, voice, seed, expected_info in specifications:
        record_path = output_dir / f"{name}.json"
        if args.resume and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            npz_path = output_dir / record["npz"]
            wav_path = output_dir / record["wav"]
            if (
                npz_path.is_file()
                and wav_path.is_file()
                and sha256(npz_path) == record["npz_sha256"]
                and sha256(wav_path) == record["wav_sha256"]
            ):
                print(f"reusing {name}", flush=True)
                cases.append(record)
                continue
        print(f"generating {name}", flush=True)
        cases.append(
            generate_case(
                engine=engine,
                output_dir=output_dir,
                name=name,
                text=text,
                language=language,
                voice=voice,
                seed=seed,
                expected_info=expected_info,
            )
        )

    text_192, raw_192 = find_exact_text_length(core, 192)
    text_193, raw_193 = find_exact_text_length(core, 193)
    medium_half = duration_and_lengths(core, m1, MEDIUM_TEXT, "en", 0.5)
    unsupported_text = "Hello \ue000"
    supported, unsupported = core.text_processor.validate_text(unsupported_text)
    chunk_size = core.base_chunk_size * core.chunk_compress_factor
    boundaries = {
        "empty_text": {"text": "", "accepted": False, "reason": "empty"},
        "unsupported_character": {
            "text": unsupported_text,
            "accepted": bool(supported),
            "unsupported": unsupported,
        },
        "text_length_192": {
            "text": text_192,
            "raw_characters": raw_192,
            "preprocessed_text_length": 192,
            "accepted_by_text_limit": True,
        },
        "text_length_193": {
            "text": text_193,
            "raw_characters": raw_193,
            "preprocessed_text_length": 193,
            "accepted_by_text_limit": False,
        },
        "latent_length_192": {
            "duration_seconds_upper_bound": 192 * chunk_size / core.sample_rate,
            "latent_length": 192,
            "accepted_by_latent_limit": True,
        },
        "latent_length_193": {
            "duration_seconds_lower_bound": (192 * chunk_size + 1) / core.sample_rate,
            "latent_length": 193,
            "accepted_by_latent_limit": False,
        },
        "medium_speed_0_5": {
            "text": MEDIUM_TEXT,
            "text_length": medium_half["text_length"],
            "latent_length": medium_half["latent_length"],
            "predicted_duration_seconds": float(medium_half["duration"][0]),
            "accepted": medium_half["text_length"] <= 192 and medium_half["latent_length"] <= 192,
        },
    }
    boundaries_path = output_dir / "boundary_cases.json"
    boundaries_path.write_text(json.dumps(boundaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "supertonic": installed_version,
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "model_dir": str(model_dir),
        "sample_rate": core.sample_rate,
        "base_chunk_size": core.base_chunk_size,
        "chunk_compress_factor": core.chunk_compress_factor,
        "latent_dimension": core.ldim * core.chunk_compress_factor,
        "source_contract": serializable_contract(),
        "compiled_contract": serializable_compiled_contract(),
        "components": component_manifest(model_dir),
        "cases": cases,
        "boundaries": boundaries,
        "boundary_cases_sha256": sha256(boundaries_path),
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "cases": cases, "boundaries": boundaries}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
