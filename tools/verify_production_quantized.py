#!/usr/bin/env python3
"""Compare the exact saved AFE model through Supertonic's production TTS loop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from model_contract import COMPILED_INPUT_SHAPES
from vector_field_utils import (
    from_afe_output,
    metrics,
    source_constants,
    style_nchw,
    to_afe_inputs,
)
from sinusoidal_inputs import (
    ROPE_INPUT,
    ROPE_TABLE_KEY,
    TIME_INPUT,
    TIME_TABLE_KEY,
    pack_rope_input,
    validate_rope_bank,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_last(value: np.ndarray, width: int) -> np.ndarray:
    if value.shape[-1] > width:
        raise ValueError(f"input width {value.shape[-1]} exceeds compiled width {width}")
    padding = [(0, 0)] * value.ndim
    padding[-1] = (0, width - value.shape[-1])
    return np.pad(value, padding, mode="constant").astype(np.float32, copy=False)


class RecordingSession:
    """Record every output while preserving the original ONNX session call."""

    def __init__(self, session: Any):
        self.session = session
        self.step_outputs: list[np.ndarray] = []

    def run(self, output_names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        outputs = self.session.run(output_names, feeds)
        self.step_outputs.append(np.asarray(outputs[0], dtype=np.float32).copy())
        return outputs


class QuantizedVectorSession:
    """Expose the extracted quantized vector field as the production ONNX session."""

    def __init__(
        self,
        model: Any,
        constants: dict[str, np.ndarray],
        width: int = 192,
        time_table: np.ndarray | None = None,
        rope_table: np.ndarray | None = None,
    ):
        if time_table is None or rope_table is None:
            raise ValueError("the final quantized graph requires timestep and RoPE tables")
        self.model = model
        self.constants = constants
        self.width = width
        self.time_table = time_table
        self.rope_table = rope_table
        self.step_outputs: list[np.ndarray] = []
        self.branch_metrics: list[dict[str, Any]] = []

    def _core_inputs(
        self,
        feeds: dict[str, np.ndarray],
        *,
        conditional: bool,
    ) -> dict[str, np.ndarray]:
        latent = pad_last(np.asarray(feeds["noisy_latent"], dtype=np.float32), self.width)
        text = pad_last(np.asarray(feeds["text_emb"], dtype=np.float32), self.width)
        latent_mask = pad_last(
            np.asarray(feeds["latent_mask"], dtype=np.float32), self.width
        )
        text_mask = pad_last(np.asarray(feeds["text_mask"], dtype=np.float32), self.width)

        if conditional:
            style_key = self.constants["conditional_style_key"]
            style_value = np.asarray(feeds["style_ttl"], dtype=np.float32)
        else:
            text = np.broadcast_to(
                self.constants["unconditional_text"], text.shape
            ).copy()
            style_key = self.constants["unconditional_style_key"]
            style_value = self.constants["unconditional_style_value"]

        core_inputs = {
            "noisy_latent": latent[:, :, None, :],
            "text_emb": text[:, :, None, :],
            "style_ttl": style_nchw(style_value),
            "style_key": style_nchw(style_key),
            "latent_mask": latent_mask[:, :, None, :],
            "text_mask": text_mask.transpose(0, 2, 1)[:, :, :, None],
        }
        step = int(np.asarray(feeds["current_step"]).reshape(-1)[0])
        if step < 0 or step >= len(self.time_table):
            raise ValueError(f"step {step} is outside time table")
        core_inputs[TIME_INPUT] = self.time_table[step]
        core_inputs[ROPE_INPUT] = pack_rope_input(
            self.rope_table,
            latent_mask,
            text_mask,
        )
        expected_shapes = dict(COMPILED_INPUT_SHAPES)
        if list(core_inputs) != list(expected_shapes):
            raise RuntimeError(f"compiled input order mismatch: {list(core_inputs)}")
        for name, expected_shape in expected_shapes.items():
            value = np.ascontiguousarray(core_inputs[name], dtype=np.float32)
            if value.shape != expected_shape:
                raise RuntimeError(f"{name} shape {value.shape} != {expected_shape}")
            core_inputs[name] = value
        return core_inputs

    def _execute_velocity(
        self, feeds: dict[str, np.ndarray], *, conditional: bool
    ) -> np.ndarray:
        outputs = list(
            self.model.execute(
                to_afe_inputs(self._core_inputs(feeds, conditional=conditional)),
                use_jax=True,
                log_level=logging.WARNING,
            )
        )
        if len(outputs) != 1:
            raise RuntimeError(f"expected one quantized output, got {len(outputs)}")
        velocity = from_afe_output(outputs[0])[:, :, 0, :]
        natural_width = feeds["noisy_latent"].shape[-1]
        return np.ascontiguousarray(velocity[:, :, :natural_width], dtype=np.float32)

    def run(self, output_names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        del output_names
        conditional = self._execute_velocity(feeds, conditional=True)
        unconditional = self._execute_velocity(feeds, conditional=False)
        latent = np.asarray(feeds["noisy_latent"], dtype=np.float32)
        latent_mask = np.asarray(feeds["latent_mask"], dtype=np.float32)
        total_step = float(np.asarray(feeds["total_step"]).reshape(-1)[0])
        updated = (
            (latent + (4.0 * conditional - 3.0 * unconditional) / total_step)
            * latent_mask
        ).astype(np.float32)
        self.step_outputs.append(updated.copy())
        return [updated]


def si_sdr(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    reference = reference - reference.mean()
    candidate = candidate - candidate.mean()
    projection = np.dot(candidate, reference) * reference / max(
        np.dot(reference, reference), np.finfo(np.float64).tiny
    )
    noise = candidate - projection
    return float(
        10.0
        * np.log10(
            max(np.dot(projection, projection), np.finfo(np.float64).tiny)
            / max(np.dot(noise, noise), np.finfo(np.float64).tiny)
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supertonic-site-packages", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-vector-estimator", type=Path, required=True)
    parser.add_argument("--quantized-dir", type=Path, required=True)
    parser.add_argument("--quantized-name", required=True)
    parser.add_argument(
        "--runtime-data",
        type=Path,
        required=True,
        help="NPZ containing the timestep table and length-indexed RoPE bank.",
    )
    parser.add_argument(
        "--reference-case",
        type=Path,
        help="Optional recorded case used to prove ONNX production reproducibility.",
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="M1")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from afe.apis.model import Model

    sys.path.append(str(args.supertonic_site_packages.resolve()))
    from supertonic import TTS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "production_quantized_vs_onnx.json"
    report: dict[str, Any] = {
        "status": "running",
        "supertonic_version": importlib.metadata.version("supertonic"),
        "seed": args.seed,
        "steps": args.steps,
        "speed": args.speed,
        "text": args.text,
        "voice": args.voice,
        "lang": args.lang,
        "artifacts": {
            "source_vector_estimator": {
                "path": str(args.source_vector_estimator.resolve()),
                "sha256": sha256(args.source_vector_estimator),
            },
            "quantized_model": {
                "path": str((args.quantized_dir / f"{args.quantized_name}.sima").resolve()),
                "sha256": sha256(args.quantized_dir / f"{args.quantized_name}.sima"),
            },
        },
    }
    runtime_data = args.runtime_data.resolve()
    report["artifacts"]["runtime_data"] = {
        "path": str(runtime_data),
        "sha256": sha256(runtime_data),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    tts = TTS(
        model="supertonic-3",
        model_dir=args.model_dir,
        auto_download=False,
        intra_op_num_threads=8,
        inter_op_num_threads=1,
    )
    style = tts.get_voice_style(args.voice)
    original_vector_session = tts.model.vector_est_ort
    onnx_session = RecordingSession(original_vector_session)
    tts.model.vector_est_ort = onnx_session
    np.random.seed(args.seed)
    onnx_wav, onnx_duration = tts.synthesize(
        args.text,
        voice_style=style,
        total_steps=args.steps,
        speed=args.speed,
        lang=args.lang,
    )

    quantized_model = Model.load(
        model_name=args.quantized_name,
        network_directory=str(args.quantized_dir),
        log_level=logging.WARNING,
        include_unquantized_net=False,
    )
    time_table = None
    rope_table = None
    with np.load(runtime_data) as archive:
        time_table = np.asarray(archive[TIME_TABLE_KEY], dtype=np.float32)
        rope_table = validate_rope_bank(archive[ROPE_TABLE_KEY])
    expected_table_shape = (args.steps, 1, 64, 1, 1)
    if time_table.shape != expected_table_shape:
        raise ValueError(
            f"time table shape {time_table.shape} != {expected_table_shape}"
        )
    quantized_session = QuantizedVectorSession(
        quantized_model,
        source_constants(args.source_vector_estimator),
        time_table=time_table,
        rope_table=rope_table,
    )
    tts.model.vector_est_ort = quantized_session
    np.random.seed(args.seed)
    quantized_wav, quantized_duration = tts.synthesize(
        args.text,
        voice_style=style,
        total_steps=args.steps,
        speed=args.speed,
        lang=args.lang,
    )

    if len(onnx_session.step_outputs) != args.steps:
        raise RuntimeError(f"ONNX production loop produced {len(onnx_session.step_outputs)} steps")
    if len(quantized_session.step_outputs) != args.steps:
        raise RuntimeError(
            f"quantized production loop produced {len(quantized_session.step_outputs)} steps"
        )

    step_metrics = []
    for index, (reference, candidate) in enumerate(
        zip(onnx_session.step_outputs, quantized_session.step_outputs), start=1
    ):
        comparison = metrics(reference, candidate)
        comparison["step"] = index
        step_metrics.append(comparison)
        print("STEP_JSON=" + json.dumps(comparison, sort_keys=True), flush=True)

    waveform_metrics = metrics(onnx_wav, quantized_wav)
    waveform_metrics["si_sdr_db"] = si_sdr(onnx_wav, quantized_wav)
    onnx_reproduction = None
    if args.reference_case:
        with np.load(args.reference_case) as archive:
            recorded_reference_wav = archive["waveform_float32"]
            recorded_reference_latent = archive["final_latent_natural"]
        onnx_reproduction = {
            "waveform": metrics(recorded_reference_wav, onnx_wav),
            "final_latent": metrics(
                recorded_reference_latent, onnx_session.step_outputs[-1]
            ),
        }
    onnx_wav_path = args.output_dir / "production_onnx.wav"
    quantized_wav_path = args.output_dir / "production_quantized.wav"
    tts.save_audio(onnx_wav, str(onnx_wav_path))
    tts.save_audio(quantized_wav, str(quantized_wav_path))

    report.update(
        status="passed",
        durations={
            "onnx": np.asarray(onnx_duration).tolist(),
            "quantized": np.asarray(quantized_duration).tolist(),
        },
        production_onnx_reproduction=onnx_reproduction,
        quantized_vs_onnx={
            "steps": step_metrics,
            "final_latent": step_metrics[-1],
            "waveform": waveform_metrics,
        },
        outputs={
            "onnx_wav": {
                "path": str(onnx_wav_path.resolve()),
                "sha256": sha256(onnx_wav_path),
            },
            "quantized_wav": {
                "path": str(quantized_wav_path.resolve()),
                "sha256": sha256(quantized_wav_path),
            },
        },
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("WAVEFORM_JSON=" + json.dumps(waveform_metrics, sort_keys=True), flush=True)
    print("REPORT=" + str(report_path.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
