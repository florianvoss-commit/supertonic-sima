#!/usr/bin/env python3
"""Verify static, padded, and optimized Supertonic vector-estimator execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    delta = candidate - reference
    reference_norm = float(np.linalg.norm(reference.reshape(-1).astype(np.float64)))
    candidate_norm = float(np.linalg.norm(candidate.reshape(-1).astype(np.float64)))
    denominator = max(reference_norm, np.finfo(np.float64).tiny)
    cosine_denominator = max(
        reference_norm * candidate_norm, np.finfo(np.float64).tiny
    )
    return {
        "shape_equal": reference.shape == candidate.shape,
        "dtype_equal": reference.dtype == candidate.dtype,
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1).astype(np.float64)) / denominator),
        "cosine": float(
            np.dot(
                reference.reshape(-1).astype(np.float64),
                candidate.reshape(-1).astype(np.float64),
            )
            / cosine_denominator
        ),
        "allclose_rtol_1e-5_atol_1e-6": bool(
            np.allclose(reference, candidate, rtol=1e-5, atol=1e-6)
        ),
        "allclose_rtol_1e-4_atol_1e-5": bool(
            np.allclose(reference, candidate, rtol=1e-4, atol=1e-5)
        ),
    }


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def source_inputs(
    arrays: dict[str, np.ndarray], latent: np.ndarray, padded: bool, step: int, total: int
) -> dict[str, np.ndarray]:
    suffix = "padded" if padded else "natural"
    return {
        "noisy_latent": latent.astype(np.float32, copy=False),
        "text_emb": arrays[f"text_emb_{suffix}"],
        "style_ttl": arrays["style_ttl"],
        "latent_mask": arrays[f"latent_mask_{suffix}"],
        "text_mask": arrays[f"text_mask_{suffix}"],
        "current_step": np.asarray([step], dtype=np.float32),
        "total_step": np.asarray([total], dtype=np.float32),
    }


def pad_latent(latent: np.ndarray, length: int = 192) -> np.ndarray:
    result = np.zeros((1, latent.shape[1], length), dtype=np.float32)
    result[..., : latent.shape[-1]] = latent
    return result


class OptimizedCore:
    def __init__(self, candidate: Path, released_source: Path):
        self.runtime = session(candidate)
        released = onnx.load(released_source, load_external_data=False)
        initializers = {
            value.name: numpy_helper.to_array(value).astype(np.float32)
            for value in released.graph.initializer
        }
        self.conditional_style_key = initializers["/vector_estimator/Expand_output_0"]
        self.unconditional_text = initializers[
            "vector_estimator.tts.ttl.uncond_masker.text_special_token"
        ]
        self.unconditional_style_key = initializers[
            "vector_estimator.tts.ttl.uncond_masker.style_key_special_token"
        ]
        self.unconditional_style_value = initializers[
            "vector_estimator.tts.ttl.uncond_masker.style_value_special_token"
        ]

    @staticmethod
    def _style_nchw(value: np.ndarray) -> np.ndarray:
        return value.transpose(0, 2, 1)[:, :, None, :]

    def velocity(
        self,
        arrays: dict[str, np.ndarray],
        latent: np.ndarray,
        step: int,
        total: int,
        conditional: bool,
    ) -> np.ndarray:
        if conditional:
            text = arrays["text_emb_padded"]
            style_key = self.conditional_style_key
            style_value = arrays["style_ttl"]
        else:
            text = np.broadcast_to(
                self.unconditional_text, arrays["text_emb_padded"].shape
            ).copy()
            style_key = self.unconditional_style_key
            style_value = self.unconditional_style_value
        inputs = {
            "noisy_latent": latent[:, :, None, :],
            "text_emb": text[:, :, None, :],
            "style_ttl": self._style_nchw(style_value),
            "style_key": self._style_nchw(style_key),
            "latent_mask": arrays["latent_mask_padded"][:, None, :, :],
            "text_mask": arrays["text_mask_padded"].transpose(0, 2, 1)[:, :, :, None],
            "current_step": np.full((1, 1, 1, 1), step, dtype=np.float32),
            "total_step": np.full((1, 1, 1, 1), total, dtype=np.float32),
        }
        return self.runtime.run(None, inputs)[0][:, :, 0, :]

    def denoise(
        self,
        arrays: dict[str, np.ndarray],
        latent: np.ndarray,
        step: int,
        total: int,
    ) -> np.ndarray:
        conditional = self.velocity(arrays, latent, step, total, True)
        unconditional = self.velocity(arrays, latent, step, total, False)
        mask = arrays["latent_mask_padded"]
        return ((latent + (4.0 * conditional - 3.0 * unconditional) / total) * mask).astype(
            np.float32
        )


def load_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def strict_pass(record: dict[str, object]) -> bool:
    return bool(
        record["shape_equal"]
        and record["dtype_equal"]
        and record["finite"]
        and record["allclose_rtol_1e-5_atol_1e-6"]
        and record["relative_l2"] <= 1e-5
        and record["cosine"] >= 0.999999
    )


def optimized_teacher_pass(record: dict[str, object]) -> bool:
    return bool(
        record["shape_equal"]
        and record["dtype_equal"]
        and record["finite"]
        and record["max_abs"] <= 1e-5
        and record["relative_l2"] <= 1e-5
        and record["cosine"] >= 0.999999
    )


def optimized_free_pass(record: dict[str, object]) -> bool:
    # Einsum/Conv changes FP32 accumulation order.  This gate remains 2,000x
    # tighter in relative L2 than the Phase-4 BF16 free-running threshold.
    return bool(
        record["shape_equal"]
        and record["dtype_equal"]
        and record["finite"]
        and record["max_abs"] <= 2e-4
        and record["relative_l2"] <= 5e-5
        and record["cosine"] >= 0.999999
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source_runtime = session(args.source)
    static_runtime = session(args.static)
    optimized = OptimizedCore(args.candidate, args.source)
    case_names = (
        "en_m1_short",
        "en_m1_medium",
        "en_f1_short",
        "es_m1_short",
        "en_m1_near_capacity",
    )
    cases = {
        name: load_case(args.reference_dir / f"{name}.npz") for name in case_names
    }

    shape_rewrite = []
    source_zero_padding_diagnostic = []
    padding = []
    for name, arrays in cases.items():
        padded_inputs = source_inputs(
            arrays, arrays["noisy_latent_padded"], True, step=0, total=8
        )
        dynamic_padded = source_runtime.run(None, padded_inputs)[0]
        static_padded = static_runtime.run(None, padded_inputs)[0]
        comparison = metrics(dynamic_padded, static_padded)
        comparison.update(case=name, passed=strict_pass(comparison))
        shape_rewrite.append(comparison)

        natural = source_runtime.run(
            None,
            source_inputs(
                arrays, arrays["noisy_latent_natural"], False, step=0, total=8
            ),
        )[0]
        cropped = dynamic_padded[..., : natural.shape[-1]]
        comparison = metrics(natural, cropped)
        tail = dynamic_padded[..., natural.shape[-1] :]
        comparison.update(
            case=name,
            padded_tail_max_abs=float(np.max(np.abs(tail))) if tail.size else 0.0,
            passed=bool(
                comparison["finite"]
                and comparison["allclose_rtol_1e-4_atol_1e-5"]
                and comparison["relative_l2"] <= 1e-4
                and comparison["cosine"] >= 0.99999
            ),
        )
        source_zero_padding_diagnostic.append(comparison)

        corrected_padded = optimized.denoise(
            arrays, arrays["noisy_latent_padded"], step=0, total=8
        )
        corrected_cropped = corrected_padded[..., : natural.shape[-1]]
        comparison = metrics(natural, corrected_cropped)
        corrected_tail = corrected_padded[..., natural.shape[-1] :]
        comparison.update(
            case=name,
            padded_tail_max_abs=(
                float(np.max(np.abs(corrected_tail))) if corrected_tail.size else 0.0
            ),
            passed=bool(
                comparison["finite"]
                and comparison["allclose_rtol_1e-4_atol_1e-5"]
                and comparison["relative_l2"] <= 1e-4
                and comparison["cosine"] >= 0.99999
            ),
        )
        padding.append(comparison)

    iterative = []
    for name in ("en_m1_short", "en_m1_medium"):
        arrays = cases[name]
        natural_length = arrays["noisy_latent_natural"].shape[-1]
        for total in (1, 2, 8):
            source_latent = arrays["noisy_latent_natural"].copy()
            teacher_steps = []
            for step in range(total):
                reference = source_runtime.run(
                    None,
                    source_inputs(arrays, source_latent, False, step, total),
                )[0]
                candidate = optimized.denoise(
                    arrays, pad_latent(source_latent), step, total
                )[..., :natural_length]
                comparison = metrics(reference, candidate)
                comparison.update(
                    step=step,
                    strict_passed=strict_pass(comparison),
                    passed=optimized_teacher_pass(comparison),
                )
                teacher_steps.append(comparison)
                source_latent = reference

            source_latent = arrays["noisy_latent_natural"].copy()
            candidate_latent = arrays["noisy_latent_padded"].copy()
            free_steps = []
            for step in range(total):
                source_latent = source_runtime.run(
                    None,
                    source_inputs(arrays, source_latent, False, step, total),
                )[0]
                candidate_latent = optimized.denoise(
                    arrays, candidate_latent, step, total
                )
                comparison = metrics(
                    source_latent, candidate_latent[..., :natural_length]
                )
                comparison.update(
                    step=step,
                    strict_passed=strict_pass(comparison),
                    passed=optimized_free_pass(comparison),
                )
                free_steps.append(comparison)
            iterative.append(
                {
                    "case": name,
                    "total_steps": total,
                    "teacher_forced": teacher_steps,
                    "free_running": free_steps,
                }
            )

    strict_records = shape_rewrite + [
        step
        for run in iterative
        for mode in ("teacher_forced", "free_running")
        for step in run[mode]
    ]
    report = {
        "models": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "static": {"path": str(args.static), "sha256": sha256(args.static)},
            "candidate": {
                "path": str(args.candidate),
                "sha256": sha256(args.candidate),
            },
        },
        "shape_rewrite": shape_rewrite,
        "released_source_zero_padding_diagnostic": source_zero_padding_diagnostic,
        "padding_masking": padding,
        "iterative": iterative,
        "summary": {
            "shape_rewrite_passed": all(item["passed"] for item in shape_rewrite),
            "padding_masking_passed": all(item["passed"] for item in padding),
            "strict_iterative_diagnostic_passed": all(
                step["strict_passed"]
                for run in iterative
                for mode in ("teacher_forced", "free_running")
                for step in run[mode]
            ),
            "optimized_iterative_passed": all(
                step["passed"]
                for run in iterative
                for mode in ("teacher_forced", "free_running")
                for step in run[mode]
            ),
            "maximum_strict_max_abs": max(float(item["max_abs"]) for item in strict_records),
            "maximum_strict_relative_l2": max(
                float(item["relative_l2"]) for item in strict_records
            ),
            "minimum_strict_cosine": min(float(item["cosine"]) for item in strict_records),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    required = (
        report["summary"]["shape_rewrite_passed"],
        report["summary"]["padding_masking_passed"],
        report["summary"]["optimized_iterative_passed"],
    )
    return 0 if all(required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
