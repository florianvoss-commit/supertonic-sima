"""Host-side inputs for the externalized Supertonic sinusoidal operators."""

from __future__ import annotations

import numpy as np


MAX_SEQUENCE_LENGTH = 192
TIME_INPUT = "time_sinusoidal"
TIME_TABLE_KEY = TIME_INPUT
TIME_INPUT_SHAPE = (1, 64, 1, 1)

ROPE_INPUT = "rope_tables"
ROPE_TABLE_KEY = "rope_sin_cos_by_length"
ROPE_PAIR_SHAPE = (1, 64, 1, MAX_SEQUENCE_LENGTH)
ROPE_INPUT_SHAPE = (1, 128, 1, MAX_SEQUENCE_LENGTH)
ROPE_BANK_SHAPE = (MAX_SEQUENCE_LENGTH, *ROPE_PAIR_SHAPE)

CONDITIONAL_STYLE_KEY = "conditional_style_key"
UNCONDITIONAL_TEXT = "unconditional_text"
UNCONDITIONAL_STYLE_KEY = "unconditional_style_key"
UNCONDITIONAL_STYLE_VALUE = "unconditional_style_value"
RUNTIME_CONSTANT_KEYS = (
    CONDITIONAL_STYLE_KEY,
    UNCONDITIONAL_TEXT,
    UNCONDITIONAL_STYLE_KEY,
    UNCONDITIONAL_STYLE_VALUE,
)


def effective_mask_length(mask: np.ndarray, *, name: str) -> int:
    """Return the number of valid positions represented by a binary mask."""

    value = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded) or np.any((rounded < 0) | (rounded > 1)):
        raise ValueError(f"{name} must be a binary 0/1 mask")
    length = int(rounded.sum())
    if length < 1 or length > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{name} effective length {length} is outside 1..{MAX_SEQUENCE_LENGTH}"
        )
    return length


def validate_rope_bank(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=np.float32)
    if table.shape != ROPE_BANK_SHAPE:
        raise ValueError(f"RoPE bank shape {table.shape} != {ROPE_BANK_SHAPE}")
    if not np.isfinite(table).all():
        raise ValueError("RoPE bank contains a non-finite value")
    return table


def pack_rope_input(
    table: np.ndarray,
    latent_mask: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray:
    """Select and pack latent/text sin-cos pairs for the static MLA graph."""

    table = validate_rope_bank(table)
    latent_length = effective_mask_length(latent_mask, name="latent_mask")
    text_length = effective_mask_length(text_mask, name="text_mask")
    packed = np.concatenate(
        [table[latent_length - 1], table[text_length - 1]], axis=1
    )
    if packed.shape != ROPE_INPUT_SHAPE:
        raise RuntimeError(f"packed RoPE shape {packed.shape} != {ROPE_INPUT_SHAPE}")
    return np.ascontiguousarray(packed, dtype=np.float32)
