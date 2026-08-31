"""Fixed source and compiler contracts for the Supertonic 3 vector field."""

from __future__ import annotations

from collections import OrderedDict


SOURCE_SHA256 = "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c"
HF_REPOSITORY = "Supertone/supertonic-3"
HF_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
SUPERTONIC_VERSION = "1.3.1"

SOURCE_INPUT_SHAPES = OrderedDict(
    [
        ("noisy_latent", (1, 144, 192)),
        ("text_emb", (1, 256, 192)),
        ("style_ttl", (1, 50, 256)),
        ("latent_mask", (1, 1, 192)),
        ("text_mask", (1, 1, 192)),
        ("current_step", (1,)),
        ("total_step", (1,)),
    ]
)
SOURCE_OUTPUT_SHAPES = OrderedDict([("denoised_latent", (1, 144, 192))])

COMPILED_INPUT_SHAPES = OrderedDict(
    [
        ("noisy_latent", (1, 144, 1, 192)),
        ("text_emb", (1, 256, 1, 192)),
        ("style_ttl", (1, 256, 1, 50)),
        ("style_key", (1, 256, 1, 50)),
        ("latent_mask", (1, 1, 1, 192)),
        # LLiMa-compatible attention score layout: [B, key_length, 1, 1].
        ("text_mask", (1, 192, 1, 1)),
        ("current_step", (1, 1, 1, 1)),
        ("total_step", (1, 1, 1, 1)),
    ]
)
COMPILED_OUTPUT_SHAPES = OrderedDict([("velocity", (1, 144, 1, 192))])

# Backward-compatible aliases used by the source staticizer/reference generator.
INPUT_SHAPES = SOURCE_INPUT_SHAPES
OUTPUT_SHAPES = SOURCE_OUTPUT_SHAPES
SYMBOLIC_DIMENSIONS = {
    "batch_size": 1,
    "text_length": 192,
    "latent_length": 192,
}
BOUNDARY_DTYPE = "float32"


def _serializable_contract(inputs: OrderedDict, outputs: OrderedDict) -> dict:
    return {
        "inputs": [
            {"index": index, "name": name, "shape": list(shape), "dtype": BOUNDARY_DTYPE}
            for index, (name, shape) in enumerate(inputs.items())
        ],
        "outputs": [
            {"index": index, "name": name, "shape": list(shape), "dtype": BOUNDARY_DTYPE}
            for index, (name, shape) in enumerate(outputs.items())
        ],
        "symbolic_dimensions": SYMBOLIC_DIMENSIONS,
    }


def serializable_contract() -> dict:
    """Return the released source-wrapper contract."""

    return _serializable_contract(SOURCE_INPUT_SHAPES, SOURCE_OUTPUT_SHAPES)


def serializable_compiled_contract() -> dict:
    """Return the Modalix all-4D vector-field-core contract."""

    return _serializable_contract(COMPILED_INPUT_SHAPES, COMPILED_OUTPUT_SHAPES)
