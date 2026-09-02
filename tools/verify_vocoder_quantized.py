#!/usr/bin/env python3
"""Compare the fixed vocoder ONNX with the exact saved AFE quantized network."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from vector_field_utils import metrics


LATENT_CHANNELS = 144
LATENT_LENGTH = 192
PCM_SAMPLES_PER_LATENT = 6 * 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_onnx_frames(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    expected = (1, 512, 1, 1152)
    if value.shape != expected:
        raise RuntimeError(f"unexpected ONNX output {value.shape}; expected {expected}")
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1)).reshape(1, -1)


def flatten_afe_frames(value: np.ndarray) -> tuple[np.ndarray, str]:
    value = np.asarray(value, dtype=np.float32)
    if value.shape == (1, 1, 1152, 512):
        return np.ascontiguousarray(value).reshape(1, -1), "NHWC"
    if value.shape == (1, 512, 1, 1152):
        return (
            np.ascontiguousarray(value.transpose(0, 2, 3, 1)).reshape(1, -1),
            "NCHW",
        )
    raise RuntimeError(f"unexpected AFE output shape: {value.shape}")


def flatten_afe_as_if_nchw(value: np.ndarray) -> np.ndarray:
    """Deliberately reinterpret the AFE NHWC buffer as NCHW for diagnosis."""
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (1, 1, 1152, 512):
        raise RuntimeError(f"unexpected AFE output shape: {value.shape}")
    nchw = value.reshape(1, 512, 1, 1152)
    return np.ascontiguousarray(nchw.transpose(0, 2, 3, 1)).reshape(1, -1)


def aggregate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "max_abs": max(item["max_abs"] for item in comparisons),
        "max_mean_abs": max(item["mean_abs"] for item in comparisons),
        "max_relative_l2": max(item["relative_l2"] for item in comparisons),
        "min_cosine": min(item["cosine"] for item in comparisons),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vocoder", type=Path, required=True)
    parser.add_argument("--surgery-vocoder", type=Path, required=True)
    parser.add_argument("--quantized-dir", type=Path, required=True)
    parser.add_argument("--quantized-name", required=True)
    parser.add_argument(
        "--weights",
        choices=("int8", "bfloat16"),
        default="int8",
        help="Weight precision used by the saved AFE model.",
    )
    parser.add_argument(
        "--reference-case", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from afe.apis.model import Model

    source_vocoder = args.source_vocoder.resolve()
    surgery_vocoder = args.surgery_vocoder.resolve()
    quantized_dir = args.quantized_dir.resolve()
    quantized_path = quantized_dir / f"{args.quantized_name}.sima"

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    source_session = ort.InferenceSession(
        source_vocoder.as_posix(), options, providers=["CPUExecutionProvider"]
    )
    surgery_session = ort.InferenceSession(
        surgery_vocoder.as_posix(), options, providers=["CPUExecutionProvider"]
    )
    quantized_model = Model.load(
        model_name=args.quantized_name,
        network_directory=str(quantized_dir),
        include_unquantized_net=False,
        log_level=logging.WARNING,
    )

    cases: list[dict[str, Any]] = []
    for case_path in args.reference_case:
        case_path = case_path.resolve()
        with np.load(case_path) as archive:
            natural_latent = np.asarray(
                archive["final_latent_natural"], dtype=np.float32
            )
        natural_length = int(natural_latent.shape[-1])
        if natural_latent.shape[:2] != (1, LATENT_CHANNELS):
            raise ValueError(f"unexpected latent shape: {natural_latent.shape}")
        if natural_length > LATENT_LENGTH:
            raise ValueError(f"latent length {natural_length} exceeds {LATENT_LENGTH}")
        padded = np.pad(
            natural_latent,
            ((0, 0), (0, 0), (0, LATENT_LENGTH - natural_length)),
        ).astype(np.float32)
        padded_4d = np.ascontiguousarray(padded[:, :, None, :])

        source_natural = source_session.run(None, {"latent": natural_latent})[0]
        surgery_full = flatten_onnx_frames(
            surgery_session.run(None, {"latent": padded_4d})[0]
        )
        outputs = list(
            quantized_model.execute(
                {"latent": np.ascontiguousarray(padded_4d.transpose(0, 2, 3, 1))},
                use_jax=True,
                log_level=logging.WARNING,
            )
        )
        if len(outputs) != 1:
            raise RuntimeError(f"expected one quantized output, got {len(outputs)}")
        quantized_full, afe_layout = flatten_afe_frames(outputs[0])
        quantized_wrong_layout = flatten_afe_as_if_nchw(outputs[0])

        natural_samples = natural_length * PCM_SAMPLES_PER_LATENT
        surgery_natural = surgery_full[:, :natural_samples]
        quantized_natural = quantized_full[:, :natural_samples]
        source_natural = np.asarray(source_natural, dtype=np.float32)
        if source_natural.shape != (1, natural_samples):
            raise RuntimeError(
                f"source waveform {source_natural.shape} != {(1, natural_samples)}"
            )

        cases.append(
            {
                "case": case_path.stem,
                "natural_latent_length": natural_length,
                "natural_waveform_samples": natural_samples,
                "afe_output_layout": afe_layout,
                "layout_hypothesis": {
                    "direct_nhwc": metrics(surgery_full, quantized_full),
                    "reinterpret_buffer_as_nchw": metrics(
                        surgery_full, quantized_wrong_layout
                    ),
                },
                "surgery_onnx_vs_quantized_full": metrics(
                    surgery_full, quantized_full
                ),
                "surgery_onnx_vs_quantized_cropped": metrics(
                    surgery_natural, quantized_natural
                ),
                "source_dynamic_vs_surgery_static_cropped": metrics(
                    source_natural, surgery_natural
                ),
                "source_dynamic_vs_quantized_cropped": metrics(
                    source_natural, quantized_natural
                ),
            }
        )
        print("CASE_JSON=" + json.dumps(cases[-1], sort_keys=True), flush=True)

    metric_names = (
        "surgery_onnx_vs_quantized_full",
        "surgery_onnx_vs_quantized_cropped",
        "source_dynamic_vs_surgery_static_cropped",
        "source_dynamic_vs_quantized_cropped",
    )
    report = {
        "status": "completed",
        "accuracy_assessment": (
            "severe_int8_weight_degradation"
            if args.weights == "int8"
            else "bfloat16_reference_comparison"
        ),
        "precision": {
            "activations": "bfloat16",
            "weights": (
                "int8_symmetric_per_channel"
                if args.weights == "int8"
                else "bfloat16"
            ),
            "boundary": "float32",
        },
        "artifacts": {
            "source_vocoder": {
                "path": str(source_vocoder),
                "sha256": sha256(source_vocoder),
            },
            "surgery_vocoder": {
                "path": str(surgery_vocoder),
                "sha256": sha256(surgery_vocoder),
            },
            "quantized_model": {
                "path": str(quantized_path),
                "sha256": sha256(quantized_path),
            },
        },
        "cases": cases,
        "aggregate": {
            name: aggregate([case[name] for case in cases]) for name in metric_names
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("SUMMARY_JSON=" + json.dumps(report["aggregate"], sort_keys=True))
    print(f"REPORT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
