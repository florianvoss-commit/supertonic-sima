"""Shared tensor helpers for Supertonic vector-field surgery and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from model_contract import (
    COMPILED_OUTPUT_SHAPES,
    PRE_EXTERNAL_INPUT_SHAPES,
)


def load_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    base = {
        "shape": list(candidate.shape),
        "shape_equal": reference.shape == candidate.shape,
        "finite": finite,
        "reference_nonfinite": int(
            np.size(reference) - np.count_nonzero(np.isfinite(reference))
        ),
        "candidate_nonfinite": int(
            np.size(candidate) - np.count_nonzero(np.isfinite(candidate))
        ),
    }
    if reference.shape != candidate.shape or not finite:
        return {
            **base,
            "max_abs": None,
            "mean_abs": None,
            "relative_l2": None,
            "cosine": None,
        }

    delta = candidate - reference
    reference64 = reference.reshape(-1).astype(np.float64)
    candidate64 = candidate.reshape(-1).astype(np.float64)
    delta64 = delta.reshape(-1).astype(np.float64)
    reference_norm = float(np.linalg.norm(reference64))
    candidate_norm = float(np.linalg.norm(candidate64))
    tiny = np.finfo(np.float64).tiny
    return {
        **base,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta64) / max(reference_norm, tiny)),
        "cosine": float(
            np.dot(reference64, candidate64)
            / max(reference_norm * candidate_norm, tiny)
        ),
    }


def source_constants(source: Path) -> dict[str, np.ndarray]:
    model = onnx.load(source, load_external_data=False)
    initializers = {
        value.name: numpy_helper.to_array(value).astype(np.float32)
        for value in model.graph.initializer
    }
    return {
        "conditional_style_key": initializers["/vector_estimator/Expand_output_0"],
        "unconditional_text": initializers[
            "vector_estimator.tts.ttl.uncond_masker.text_special_token"
        ],
        "unconditional_style_key": initializers[
            "vector_estimator.tts.ttl.uncond_masker.style_key_special_token"
        ],
        "unconditional_style_value": initializers[
            "vector_estimator.tts.ttl.uncond_masker.style_value_special_token"
        ],
    }


def style_nchw(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 1)[:, :, None, :], dtype=np.float32)


def pre_external_inputs(
    arrays: dict[str, np.ndarray],
    constants: dict[str, np.ndarray],
    latent: np.ndarray,
    step: int,
    total: int,
    conditional: bool,
) -> dict[str, np.ndarray]:
    """Build inputs for the internal graph before Sin/Cos externalization."""

    if conditional:
        text = arrays["text_emb_padded"]
        style_key = constants["conditional_style_key"]
        style_value = arrays["style_ttl"]
    else:
        text = np.broadcast_to(
            constants["unconditional_text"], arrays["text_emb_padded"].shape
        ).copy()
        style_key = constants["unconditional_style_key"]
        style_value = constants["unconditional_style_value"]

    feeds = {
        "noisy_latent": latent[:, :, None, :],
        "text_emb": text[:, :, None, :],
        "style_ttl": style_nchw(style_value),
        "style_key": style_nchw(style_key),
        "latent_mask": arrays["latent_mask_padded"][:, None, :, :],
        "text_mask": arrays["text_mask_padded"].transpose(0, 2, 1)[:, :, :, None],
        "current_step": np.full((1, 1, 1, 1), step, dtype=np.float32),
        "total_step": np.full((1, 1, 1, 1), total, dtype=np.float32),
    }
    if list(feeds) != list(PRE_EXTERNAL_INPUT_SHAPES):
        raise RuntimeError(f"pre-external input order mismatch: {list(feeds)}")
    for name, expected_shape in PRE_EXTERNAL_INPUT_SHAPES.items():
        value = np.ascontiguousarray(feeds[name], dtype=np.float32)
        if value.shape != expected_shape:
            raise RuntimeError(f"{name} shape {value.shape} != {expected_shape}")
        feeds[name] = value
    return feeds


def to_afe_inputs(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert the ordered ONNX NCHW contract to AFE's NHWC boundary."""

    return {
        name: np.ascontiguousarray(value.transpose(0, 2, 3, 1))
        for name, value in feeds.items()
    }


def from_afe_output(value: np.ndarray) -> np.ndarray:
    expected = next(iter(COMPILED_OUTPUT_SHAPES.values()))
    value = np.asarray(value, dtype=np.float32)
    if value.shape == expected:
        return value
    channel_last = (expected[0], expected[2], expected[3], expected[1])
    if value.shape == channel_last:
        return value.transpose(0, 3, 1, 2)
    raise RuntimeError(
        f"unexpected AFE output shape {value.shape}; expected {expected} or {channel_last}"
    )
