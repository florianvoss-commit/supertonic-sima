#!/usr/bin/env python3
"""Implementation details for the Supertonic vector-estimator surgery."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference

from contract import (
    COMPILED_INPUT_SHAPES,
    COMPILED_OUTPUT_SHAPES,
    HF_REPOSITORY,
    HF_REVISION,
    INPUT_SHAPES,
    OUTPUT_SHAPES,
    PRE_EXTERNAL_INPUT_SHAPES,
    SOURCE_INPUT_SHAPES,
    SOURCE_SHA256,
    SUPERTONIC_VERSION,
    SYMBOLIC_DIMENSIONS,
    serializable_compiled_contract,
    serializable_contract,
    serializable_pre_external_contract,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "supertonic_sima"))
from inputs import (  # noqa: E402
    MAX_SEQUENCE_LENGTH,
    ROPE_BANK_SHAPE,
    ROPE_INPUT,
    ROPE_INPUT_SHAPE,
    ROPE_TABLE_KEY,
    TIME_INPUT,
    TIME_INPUT_SHAPE,
    pack_rope_input,
)


REPLACEMENTS = {
    "/vector_estimator/Tile_1_output_0": "noisy_latent",
    "/vector_estimator/Tile_3_output_0": "latent_mask",
    "/vector_estimator/Tile_4_output_0": "text_mask",
    "/vector_estimator/Tile_2_output_0": "/vector_estimator/Expand_5_output_0",
    "/vector_estimator/Concat_5_output_0": "text_emb",
    "/vector_estimator/Concat_6_output_0": "style_key",
    "/vector_estimator/Concat_7_output_0": "style_ttl",
}

RAW_VELOCITY = "/vector_estimator/vector_field/proj_out/Mul_output_0"


# Batch-one vector-field extraction.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_to_output(model: onnx.ModelProto, output_name: str) -> None:
    """Remove nodes and initializers that cannot affect *output_name*."""

    producer = {out: node for node in model.graph.node for out in node.output}
    required_tensors = {output_name}
    required_node_ids: set[int] = set()
    stack = [output_name]
    while stack:
        tensor = stack.pop()
        node = producer.get(tensor)
        if node is None or id(node) in required_node_ids:
            continue
        required_node_ids.add(id(node))
        for node_input in node.input:
            if node_input and node_input not in required_tensors:
                required_tensors.add(node_input)
                stack.append(node_input)

    kept_nodes = [node for node in model.graph.node if id(node) in required_node_ids]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    kept_initializers = [
        value for value in model.graph.initializer if value.name in required_tensors
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    initializer_names = {value.name for value in kept_initializers}
    used_graph_inputs = required_tensors - initializer_names
    kept_inputs = [value for value in model.graph.input if value.name in used_graph_inputs]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)

    # The original inferred value-info records the wrapper's internal batch of
    # two.  Discard it and infer fresh batch-1 metadata after rewiring.
    kept_value_info: list[onnx.ValueInfoProto] = []
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)


def extract(source: Path, output: Path) -> dict[str, object]:
    model = onnx.load(source)

    # The released wrapper owns the conditional and unconditional style-key
    # contexts.  Make the selected key explicit so one batch-1 core can execute
    # either branch without retaining an internal CFG batch.
    ordered_inputs = list(model.graph.input)
    ordered_inputs.insert(
        3,
        helper.make_tensor_value_info(
            "style_key", TensorProto.FLOAT, [1, 50, 256]
        ),
    )
    del model.graph.input[:]
    model.graph.input.extend(ordered_inputs)

    replacement_hits = {name: 0 for name in REPLACEMENTS}
    for node in model.graph.node:
        for index, node_input in enumerate(node.input):
            replacement = REPLACEMENTS.get(node_input)
            if replacement is not None:
                node.input[index] = replacement
                replacement_hits[node_input] += 1

    missing = [name for name, hits in replacement_hits.items() if hits == 0]
    if missing:
        raise RuntimeError(f"expected tensors were not consumed: {missing}")

    identity = helper.make_node(
        "Identity", [RAW_VELOCITY], ["velocity"], name="/modalix/velocity"
    )
    model.graph.node.append(identity)
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("velocity", TensorProto.FLOAT, [1, 144, 192])
    )
    prune_to_output(model, "velocity")

    model.graph.name = "supertonic_vector_field_b1_t192_l192"
    model.doc_string = (
        "Batch-1 Supertonic vector field extracted from the released CFG/Euler "
        "wrapper. The caller performs two evaluations and applies 4*c-3*u."
    )
    checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    checker.check_model(inferred, full_check=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)
    reloaded = onnx.load(output)
    checker.check_model(reloaded, full_check=True)

    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "nodes": len(reloaded.graph.node),
        "initializers": len(reloaded.graph.initializer),
        "replacement_consumers": replacement_hits,
        "inputs": [value.name for value in reloaded.graph.input],
        "output_name": reloaded.graph.output[0].name,
        "output_shape": [1, 144, 192],
        "guidance_formula": "4 * conditional_velocity - 3 * unconditional_velocity",
    }


# Source graph staticization.


def tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    dims: list[int | str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(dim.dim_param or "?")
    return tuple(dims)


def describe_graph(model: onnx.ModelProto) -> dict:
    return {
        "inputs": [
            {
                "index": index,
                "name": value.name,
                "shape": list(tensor_shape(value)),
                "onnx_element_type": int(value.type.tensor_type.elem_type),
            }
            for index, value in enumerate(model.graph.input)
        ],
        "outputs": [
            {
                "index": index,
                "name": value.name,
                "shape": list(tensor_shape(value)),
                "onnx_element_type": int(value.type.tensor_type.elem_type),
            }
            for index, value in enumerate(model.graph.output)
        ],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "ir_version": int(model.ir_version),
        "opsets": [
            {"domain": opset.domain, "version": int(opset.version)}
            for opset in model.opset_import
        ],
    }


def freeze_symbolic_dimensions(model: onnx.ModelProto) -> dict[str, int]:
    changed = {name: 0 for name in SYMBOLIC_DIMENSIONS}
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim in tensor_type.shape.dim:
            symbolic = dim.dim_param
            if symbolic in SYMBOLIC_DIMENSIONS:
                dim.ClearField("dim_param")
                dim.dim_value = SYMBOLIC_DIMENSIONS[symbolic]
                changed[symbolic] += 1
    return changed


def validate_contract(model: onnx.ModelProto) -> None:
    actual_inputs = [(value.name, tensor_shape(value)) for value in model.graph.input]
    actual_outputs = [(value.name, tensor_shape(value)) for value in model.graph.output]
    expected_inputs = list(INPUT_SHAPES.items())
    expected_outputs = list(OUTPUT_SHAPES.items())
    if actual_inputs != expected_inputs:
        raise ValueError(f"input contract mismatch: expected {expected_inputs}, got {actual_inputs}")
    if actual_outputs != expected_outputs:
        raise ValueError(f"output contract mismatch: expected {expected_outputs}, got {actual_outputs}")
    for value in list(model.graph.input) + list(model.graph.output):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"{value.name} is not an FP32 ONNX tensor")


def simplify_model(static_path: Path, simplified_path: Path) -> dict:
    executable = shutil.which("onnxsim")
    if executable is None:
        raise RuntimeError("onnxsim is not installed in the active environment")
    completed = subprocess.run(
        [executable, str(static_path), str(simplified_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    model = onnx.load(str(simplified_path))
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    validate_contract(model)
    onnx.save(model, str(simplified_path))
    return {"tool_output": completed.stdout, "graph": describe_graph(model)}


def staticize(argv: list[str] | None = None, *, emit: bool = True) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--simplified-output", type=Path)
    args = parser.parse_args(argv)

    source = args.input.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    simplified_output = args.simplified_output.resolve() if args.simplified_output else None
    if not source.is_file():
        raise FileNotFoundError(source)

    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256:
        raise ValueError(f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {source_hash}")

    model = onnx.load(str(source))
    source_graph = describe_graph(model)
    changed = freeze_symbolic_dimensions(model)
    if any(count == 0 for count in changed.values()):
        raise ValueError(f"not every required symbolic dimension was found: {changed}")

    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    validate_contract(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))

    # Reload the serialized artifact so validation covers the exact file on disk.
    serialized = onnx.load(str(output))
    onnx.checker.check_model(serialized, full_check=True)
    validate_contract(serialized)

    result = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "onnx": onnx.__version__,
        "source": {
            "path": str(source),
            "sha256": source_hash,
            "hf_repository": HF_REPOSITORY,
            "hf_revision": HF_REVISION,
            "supertonic_version": SUPERTONIC_VERSION,
            "graph": source_graph,
        },
        "static": {
            "path": str(output),
            "sha256": sha256(output),
            "frozen_dimension_occurrences": changed,
            "graph": describe_graph(serialized),
        },
        "expected_contract": serializable_contract(),
    }

    if simplified_output is not None:
        simplified_output.parent.mkdir(parents=True, exist_ok=True)
        simplified = simplify_model(output, simplified_output)
        simplified.update({"path": str(simplified_output), "sha256": sha256(simplified_output)})
        result["simplified"] = simplified

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if emit:
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


# Reference-case and tensor-layout helpers.


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
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


# Rank-four lifting.


PARAMETER_INPUTS: dict[str, tuple[int, ...]] = {
    "Reshape": (1,),
    "Expand": (1,),
    "Tile": (1,),
    "Pad": (1,),
    "Unsqueeze": (1,),
    "Squeeze": (1,),
    "Slice": (1, 2, 3, 4),
    "Split": (1,),
    "ReduceSum": (1,),
    "ConstantOfShape": (0,),
}


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(1)
    return result


def fixed(shape: list[int | str]) -> list[int]:
    # Remaining unknowns in this fully static graph are conservative inference
    # artifacts for the batch dimension.
    return [1 if not isinstance(value, int) or value <= 0 else value for value in shape]


def classify(name: str, shape: list[int]) -> str:
    rank = len(shape)
    if rank == 4:
        if (shape[0] != 1 and shape[1] == 1) or "/attention/" in name:
            return "head_first"
        return "attention"
    if rank == 3:
        if shape[1] in {1, 64, 144, 256, 512, 2048} and shape[-1] in {1, 192}:
            return "channel_first"
        return "channel_last"
    if rank == 2:
        return "batch_feature"
    if rank == 1:
        return "batch_scalar"
    if rank == 0:
        return "scalar"
    raise ValueError(f"unsupported activation rank {rank} for {name}: {shape}")


def physical_shape(layout: str, shape: list[int]) -> list[int]:
    if layout == "channel_first":
        return [shape[0], shape[1], 1, shape[2]]
    if layout == "channel_last":
        return [shape[0], 1, shape[1], shape[2]]
    if layout == "batch_feature":
        return [shape[0], 1, 1, shape[1]]
    if layout == "batch_scalar":
        return [shape[0], 1, 1, 1]
    if layout == "scalar":
        return [1, 1, 1, 1]
    if layout == "head_first":
        return [shape[1], shape[0], shape[2], shape[3]]
    if layout == "attention":
        return shape
    raise ValueError(layout)


def embedding(layout: str, rank: int) -> dict[int, int]:
    if layout == "channel_first":
        return {0: 0, 1: 1, 2: 3}
    if layout == "channel_last":
        return {0: 0, 1: 2, 2: 3}
    if layout == "batch_feature":
        return {0: 0, 1: 3}
    if layout == "batch_scalar":
        return {0: 0}
    if layout == "scalar":
        return {}
    if layout == "head_first":
        return {0: 1, 1: 0, 2: 2, 3: 3}
    if layout == "attention":
        return {axis: axis for axis in range(rank)}
    raise ValueError(layout)


def transpose_perm(
    old_perm: list[int], in_layout: str, out_layout: str, in_rank: int, out_rank: int
) -> list[int]:
    in_embed = embedding(in_layout, in_rank)
    out_embed = embedding(out_layout, out_rank)
    result: list[int | None] = [None] * 4
    used: set[int] = set()
    for old_out_axis, old_in_axis in enumerate(old_perm):
        new_out_axis = out_embed[old_out_axis]
        new_in_axis = in_embed[old_in_axis]
        result[new_out_axis] = new_in_axis
        used.add(new_in_axis)
    spare_inputs = [axis for axis in range(4) if axis not in used]
    for axis in range(4):
        if result[axis] is None:
            result[axis] = spare_inputs.pop(0)
    return [int(axis) for axis in result]


def attr_ints(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None


def attr_int(node: onnx.NodeProto, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def set_ints(node: onnx.NodeProto, name: str, values: list[int]) -> None:
    kept = [copy.deepcopy(attr) for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, values))


def set_int(node: onnx.NodeProto, name: str, value: int) -> None:
    kept = [copy.deepcopy(attr) for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def make_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> str:
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))
    return name


def control_tensors(model: onnx.ModelProto) -> set[str]:
    """Find tensors used exclusively as operator shape/index parameters."""

    producer = {out: node for node in model.graph.node for out in node.output}
    control: set[str] = set()
    stack: list[str] = []
    for node in model.graph.node:
        for index in PARAMETER_INPUTS.get(node.op_type, ()):
            if index < len(node.input) and node.input[index]:
                stack.append(node.input[index])
    while stack:
        name = stack.pop()
        if name in control:
            continue
        control.add(name)
        node = producer.get(name)
        if node is None or node.op_type == "Shape":
            continue
        stack.extend(value for value in node.input if value)
    return control


def fold_forbidden_shape_ops(model: onnx.ModelProto) -> Counter:
    """Fold static Gather/Unsqueeze/Squeeze nodes to Constant nodes."""

    values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    folded = Counter()
    for node in model.graph.node:
        result: np.ndarray | None = None
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    result = numpy_helper.to_array(attr.t)
                    break
        elif node.op_type == "Gather" and all(name in values for name in node.input):
            axis = attr_int(node, "axis", 0)
            result = np.take(values[node.input[0]], values[node.input[1]], axis=axis)
        elif node.op_type in {"Unsqueeze", "Squeeze"} and all(
            name in values for name in node.input
        ):
            axes = [int(axis) for axis in values[node.input[1]].reshape(-1)]
            result = values[node.input[0]]
            if node.op_type == "Unsqueeze":
                for axis in sorted(axes):
                    result = np.expand_dims(result, axis=axis)
            else:
                result = np.squeeze(result, axis=tuple(axes))

        if result is None:
            continue
        values[node.output[0]] = np.asarray(result)
        if node.op_type in {"Gather", "Unsqueeze", "Squeeze"}:
            old_op = node.op_type
            del node.input[:]
            del node.attribute[:]
            node.op_type = "Constant"
            node.attribute.append(
                helper.make_attribute(
                    "value", numpy_helper.from_array(np.asarray(result))
                )
            )
            folded[old_op] += 1
    return folded


def prune_unused(model: onnx.ModelProto) -> dict[str, int]:
    """Prune dead static-shape subgraphs and their initializers."""

    old_nodes = len(model.graph.node)
    old_initializers = len(model.graph.initializer)
    producer = {out: node for node in model.graph.node for out in node.output}
    required = {value.name for value in model.graph.output}
    required_node_ids: set[int] = set()
    stack = list(required)
    while stack:
        name = stack.pop()
        node = producer.get(name)
        if node is None or id(node) in required_node_ids:
            continue
        required_node_ids.add(id(node))
        for node_input in node.input:
            if node_input and node_input not in required:
                required.add(node_input)
                stack.append(node_input)

    nodes = [node for node in model.graph.node if id(node) in required_node_ids]
    initializers = [
        value for value in model.graph.initializer if value.name in required
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return {
        "nodes": old_nodes - len(nodes),
        "initializers": old_initializers - len(initializers),
    }


def rewrite_equal_where_masks(model: onnx.ModelProto) -> Counter:
    """Replace binary-mask Equal/Where patterns with arithmetic masking.

    Pre-softmax key masking becomes
    ``scores + ((1 - mask) * finfo(float32).min)``.  For a binary mask this is
    exactly 0 for valid keys and the smallest finite FP32 value for invalid
    keys.  Keeping the fill finite avoids NaNs in AFE evaluators and BF16
    lowering while still underflowing masked Softmax probabilities to zero.
    Post-softmax query masking becomes a direct multiply by the binary mask.
    """

    producer = {out: node for node in model.graph.node for out in node.output}
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    one_name = "modalix.mask_one"
    if one_name not in initializers:
        make_initializer(model, one_name, np.asarray(1.0, dtype=np.float32))
    fill_name = "modalix.mask_fill_fp32_min"
    if fill_name not in initializers:
        make_initializer(
            model, fill_name, np.asarray(np.finfo(np.float32).min, dtype=np.float32)
        )

    new_nodes: list[onnx.NodeProto] = []
    rewritten = Counter()
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Where":
            new_nodes.append(node)
            continue
        equal = producer.get(node.input[0])
        if equal is None or equal.op_type != "Equal":
            raise RuntimeError(f"unsupported non-mask Where node: {node.name}")
        mask = equal.input[0]
        true_value = initializers.get(node.input[1])
        if true_value is None:
            raise RuntimeError(f"non-constant true branch in {node.name}")
        scalar = float(np.asarray(true_value).reshape(()))
        data = node.input[2]
        if scalar == 0.0:
            new_nodes.append(
                helper.make_node("Mul", [data, mask], list(node.output), name=node.name)
            )
            rewritten["Where->Mul"] += 1
        elif np.isneginf(scalar):
            inverted = f"{node.output[0]}/modalix_mask_inverted"
            bias = f"{node.output[0]}/modalix_mask_bias"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Sub",
                        [one_name, mask],
                        [inverted],
                        name=f"{node.name}/modalix_mask_invert",
                    ),
                    helper.make_node(
                        "Mul",
                        [inverted, fill_name],
                        [bias],
                        name=f"{node.name}/modalix_mask_bias",
                    ),
                    helper.make_node("Add", [data, bias], list(node.output), name=node.name),
                ]
            )
            rewritten["Where->FiniteMinMask"] += 1
        else:
            raise RuntimeError(f"unsupported Where fill {scalar} in {node.name}")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return rewritten


def lift(source: Path, output: Path) -> dict[str, object]:
    original = shape_inference.infer_shapes(
        onnx.load(source), strict_mode=True, data_prop=True
    )
    model = copy.deepcopy(original)
    old_shapes = {
        value.name: fixed(dims(value))
        for value in list(original.graph.input)
        + list(original.graph.value_info)
        + list(original.graph.output)
        if value.type.tensor_type.HasField("shape")
    }
    controls = control_tensors(original)
    initializer_map = {value.name: value for value in model.graph.initializer}
    initializer_names = set(initializer_map)
    old_initializer_shapes = {
        value.name: list(numpy_helper.to_array(value).shape)
        for value in original.graph.initializer
    }

    def is_activation(name: str) -> bool:
        return bool(name) and name not in controls and name not in initializer_names

    layouts = {
        name: classify(name, shape)
        for name, shape in old_shapes.items()
        if is_activation(name)
    }
    physical = {
        name: physical_shape(layouts[name], shape)
        for name, shape in old_shapes.items()
        if name in layouts
    }

    # Public inputs and outputs use the new physical contract.
    for value in list(model.graph.input) + list(model.graph.output):
        if value.name not in physical:
            continue
        del value.type.tensor_type.shape.dim[:]
        for size in physical[value.name]:
            value.type.tensor_type.shape.dim.add().dim_value = size

    # Reshape non-parameter rank-3 initializers used in broadcast arithmetic.
    conv_weights = {
        node.input[1]
        for node in model.graph.node
        if node.op_type == "Conv" and len(node.input) > 1
    }
    pad_vectors = {
        node.input[1]
        for node in model.graph.node
        if node.op_type == "Pad" and len(node.input) > 1
    }
    for name, value in list(initializer_map.items()):
        array = numpy_helper.to_array(value)
        if name in conv_weights and array.ndim == 3:
            replacement = array[:, :, None, :]
        elif name in pad_vectors and array.size == 6:
            p = array.reshape(-1)
            replacement = np.asarray(
                [p[0], p[1], 0, p[2], p[3], p[4], 0, p[5]], dtype=p.dtype
            )
        elif array.ndim == 3 and not name.startswith("onnx::"):
            layout = classify(name, list(array.shape))
            replacement = array.reshape(physical_shape(layout, list(array.shape)))
        else:
            continue
        value.CopyFrom(numpy_helper.from_array(replacement, name=name))

    new_nodes: list[onnx.NodeProto] = []
    rewritten = Counter()
    for node_index, node in enumerate(model.graph.node):
        node = copy.deepcopy(node)
        data_inputs = [name for name in node.input if is_activation(name)]
        data_outputs = [name for name in node.output if is_activation(name)]

        if node.op_type == "Shape" and node.input[0] in old_shapes:
            old = np.asarray(old_shapes[node.input[0]], dtype=np.int64)
            node.op_type = "Constant"
            del node.input[:]
            del node.attribute[:]
            node.attribute.append(
                helper.make_attribute("value", numpy_helper.from_array(old))
            )
            rewritten["Shape->Constant"] += 1

        elif node.op_type == "Conv":
            for attr_name, default in (
                ("dilations", [1]),
                ("kernel_shape", None),
                ("strides", [1]),
            ):
                values = attr_ints(node, attr_name)
                if values is None:
                    values = default
                if values is not None and len(values) == 1:
                    set_ints(node, attr_name, [1, values[0]])
            pads = attr_ints(node, "pads")
            if pads is not None and len(pads) == 2:
                set_ints(node, "pads", [0, pads[0], 0, pads[1]])
            rewritten["Conv1D->Conv2D"] += 1

        elif node.op_type == "Gemm":
            weight_name = node.input[1]
            if attr_int(node, "transB") != 0:
                weight = initializer_map.get(weight_name)
                if weight is None:
                    raise RuntimeError(f"non-constant transB in {node.name}")
                weight_name = f"modalix.gemm_weight.{node_index}"
                make_initializer(
                    model,
                    weight_name,
                    numpy_helper.to_array(weight).T.copy(),
                )
            matmul_out = f"{node.output[0]}/modalix_matmul"
            new_nodes.append(
                helper.make_node(
                    "MatMul", [node.input[0], weight_name], [matmul_out],
                    name=f"{node.name}/modalix_matmul",
                )
            )
            node = helper.make_node(
                "Add", [matmul_out, node.input[2]], list(node.output), name=node.name
            )
            rewritten["Gemm->MatMulAdd"] += 1

        elif node.op_type == "Transpose" and data_inputs and data_outputs:
            old_perm = attr_ints(node, "perm")
            if old_perm is None:
                old_perm = list(reversed(range(len(old_shapes[data_inputs[0]]))))
            set_ints(
                node,
                "perm",
                transpose_perm(
                    old_perm,
                    layouts[data_inputs[0]],
                    layouts[data_outputs[0]],
                    len(old_shapes[data_inputs[0]]),
                    len(old_shapes[data_outputs[0]]),
                ),
            )
            rewritten["Transpose"] += 1

        elif node.op_type == "Reshape" and data_outputs:
            target = np.asarray(physical[data_outputs[0]], dtype=np.int64)
            shape_name = f"modalix.reshape_shape.{node_index}"
            make_initializer(model, shape_name, target)
            node.input[1] = shape_name
            rewritten["Reshape"] += 1

        elif node.op_type in {"Unsqueeze", "Squeeze"} and data_outputs:
            # Rank adaptation is redundant because both sides now have their
            # permanent four-dimensional representation.
            if data_inputs and int(np.prod(physical[data_inputs[0]])) == int(
                np.prod(physical[data_outputs[0]])
            ):
                old_op = node.op_type
                if physical[data_inputs[0]] == physical[data_outputs[0]]:
                    node = helper.make_node(
                        "Identity", [data_inputs[0]], list(node.output), name=node.name
                    )
                else:
                    target_name = f"modalix.rank4_shape.{node_index}"
                    make_initializer(
                        model,
                        target_name,
                        np.asarray(physical[data_outputs[0]], dtype=np.int64),
                    )
                    node = helper.make_node(
                        "Reshape",
                        [data_inputs[0], target_name],
                        list(node.output),
                        name=node.name,
                    )
                rewritten[f"{old_op}->rank4"] += 1

        elif node.op_type in {"Concat", "Split", "Softmax"} and data_inputs:
            old_axis = attr_int(node, "axis", 0)
            in_name = data_inputs[0]
            rank = len(old_shapes[in_name])
            if old_axis < 0:
                old_axis += rank
            new_axis = embedding(layouts[in_name], rank)[old_axis]
            set_int(node, "axis", new_axis)
            rewritten[node.op_type] += 1

        elif node.op_type == "Slice" and (
            data_inputs
            or (
                node.input[0] in old_initializer_shapes
                and len(old_initializer_shapes[node.input[0]]) >= 2
            )
        ):
            axes_name = node.input[3] if len(node.input) > 3 else ""
            axes_value = initializer_map.get(axes_name)
            if axes_value is not None:
                old_axes = numpy_helper.to_array(axes_value).astype(np.int64)
                source_name = data_inputs[0] if data_inputs else node.input[0]
                source_shape = (
                    old_shapes[source_name]
                    if source_name in old_shapes
                    else old_initializer_shapes[source_name]
                )
                source_layout = (
                    layouts[source_name]
                    if source_name in layouts
                    else classify(source_name, source_shape)
                )
                rank = len(source_shape)
                mapped = [
                    embedding(source_layout, rank)[
                        int(axis if axis >= 0 else axis + rank)
                    ]
                    for axis in old_axes
                ]
                axes_new = f"modalix.slice_axes.{node_index}"
                make_initializer(model, axes_new, np.asarray(mapped, dtype=np.int64))
                node.input[3] = axes_new
                rewritten["Slice"] += 1

        elif node.op_type == "ReduceSum" and data_inputs and data_outputs:
            axes_value = initializer_map.get(node.input[1])
            if axes_value is not None:
                old_rank = len(old_shapes[data_inputs[0]])
                old_axes = [
                    int(axis if axis >= 0 else axis + old_rank)
                    for axis in numpy_helper.to_array(axes_value).reshape(-1)
                ]
                mapped = [embedding(layouts[data_inputs[0]], old_rank)[a] for a in old_axes]
                # Reduce inserted singleton dimensions as well and retain four
                # dimensions in the result.
                mapped.extend(
                    axis
                    for axis in range(4)
                    if axis not in embedding(layouts[data_inputs[0]], old_rank).values()
                )
                axes_new = f"modalix.reduce_axes.{node_index}"
                make_initializer(
                    model, axes_new, np.asarray(sorted(set(mapped)), dtype=np.int64)
                )
                node.input[1] = axes_new
                set_int(node, "keepdims", 1)
                rewritten["ReduceSum"] += 1

        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    rewritten.update(fold_forbidden_shape_ops(model))
    rewritten.update(rewrite_equal_where_masks(model))
    pruned = prune_unused(model)
    del model.graph.value_info[:]
    model.graph.name = "supertonic_vector_field_b1_t192_l192_all4d"
    model.doc_string = (
        "Batch-1 Supertonic vector field with rank-4 public tensors and data "
        "activations. Shape/index bookkeeping tensors retain natural rank."
    )

    checker.check_model(model)
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception:
        invalid_path = output.with_suffix(".invalid.onnx")
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, invalid_path)
        raise
    checker.check_model(inferred, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)
    reloaded = onnx.load(output)
    checker.check_model(reloaded, full_check=True)

    ranks = Counter()
    violations: list[dict[str, object]] = []
    inferred_controls = control_tensors(reloaded)
    inferred_initializers = {value.name for value in reloaded.graph.initializer}
    for value in list(reloaded.graph.input) + list(reloaded.graph.value_info) + list(
        reloaded.graph.output
    ):
        shape = dims(value)
        ranks[len(shape)] += 1
        if (
            value.name not in inferred_controls
            and value.name not in inferred_initializers
            and (len(shape) != 4 or fixed(shape)[0] != 1)
        ):
            violations.append({"name": value.name, "shape": shape})

    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "nodes": len(reloaded.graph.node),
        "initializers": len(reloaded.graph.initializer),
        "rewritten": dict(rewritten),
        "pruned": pruned,
        "rank_histogram": dict(sorted(ranks.items())),
        "activation_rank_violations": violations,
        "unsqueeze_nodes": sum(n.op_type == "Unsqueeze" for n in reloaded.graph.node),
        "squeeze_nodes": sum(n.op_type == "Squeeze" for n in reloaded.graph.node),
        "gather_nodes": sum(n.op_type == "Gather" for n in reloaded.graph.node),
        "equal_nodes": sum(n.op_type == "Equal" for n in reloaded.graph.node),
        "where_nodes": sum(n.op_type == "Where" for n in reloaded.graph.node),
    }




# MLA-oriented graph optimization.


def shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    return {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }


def perm(node: onnx.NodeProto) -> tuple[int, ...]:
    for attr in node.attribute:
        if attr.name == "perm":
            return tuple(attr.ints)
    return ()


def consumers(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for value in node.input:
            result.setdefault(value, []).append(node)
    return result


def replace_inputs(
    model: onnx.ModelProto, old: str, new: str, skip_name: str | None = None
) -> None:
    for node in model.graph.node:
        if node.name == skip_name:
            continue
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def remove_identities(model: onnx.ModelProto) -> int:
    output_names = {value.name for value in model.graph.output}
    producer = {out: node for node in model.graph.node for out in node.output}
    remove_names: set[str] = set()
    count = 0
    for node in model.graph.node:
        if node.op_type != "Identity":
            continue
        source, target = node.input[0], node.output[0]
        if target in output_names:
            source_node = producer.get(source)
            if source_node is None:
                raise RuntimeError(f"cannot preserve graph output {target}")
            for index, value in enumerate(source_node.output):
                if value == source:
                    source_node.output[index] = target
            replace_inputs(model, source, target, skip_name=node.name)
        else:
            replace_inputs(model, target, source, skip_name=node.name)
        remove_names.add(node.name)
        count += 1
    kept = [node for node in model.graph.node if node.name not in remove_names]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return count


def remove_noop_reshapes(model: onnx.ModelProto) -> int:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes = shape_map(inferred)
    remove_names: set[str] = set()
    count = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        if shapes.get(node.input[0]) != shapes.get(node.output[0]):
            continue
        replace_inputs(model, node.output[0], node.input[0], skip_name=node.name)
        remove_names.add(node.name)
        count += 1
    kept = [node for node in model.graph.node if node.name not in remove_names]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return count


def remove_singleton_reshape_transpose_fanouts(model: onnx.ModelProto) -> Counter:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes = shape_map(inferred)
    uses = consumers(model)
    remove_names: set[str] = set()
    result = Counter()
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        source_shape = shapes.get(node.input[0])
        middle_shape = shapes.get(node.output[0])
        fanout = uses.get(node.output[0], [])
        if not source_shape or not middle_shape or not fanout:
            continue
        if sum(size != 1 for size in middle_shape) > 1:
            continue
        if not all(
            child.op_type == "Transpose"
            and shapes.get(child.output[0]) == source_shape
            for child in fanout
        ):
            continue
        for child in fanout:
            replace_inputs(model, child.output[0], node.input[0], skip_name=child.name)
            remove_names.add(child.name)
            result["Transpose"] += 1
        remove_names.add(node.name)
        result["Reshape"] += 1
    kept = [node for node in model.graph.node if node.name not in remove_names]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return result


def replace_head_pack_and_merge(model: onnx.ModelProto) -> Counter:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes = shape_map(inferred)
    uses = consumers(model)
    producer = {out: node for node in model.graph.node for out in node.output}
    replacements: dict[str, list[onnx.NodeProto]] = {}
    skip: set[str] = set()
    result = Counter()
    serial = 0

    # [B,1,L,H*D] -> reshape [B,L,H,D] -> transpose [B,H,L,D]
    for reshape in model.graph.node:
        if reshape.op_type != "Reshape" or reshape.name in skip:
            continue
        children = uses.get(reshape.output[0], [])
        if len(children) != 1 or children[0].op_type != "Transpose":
            continue
        transpose = children[0]
        source = shapes.get(reshape.input[0])
        middle = shapes.get(reshape.output[0])
        target = shapes.get(transpose.output[0])
        if (
            not source
            or not middle
            or not target
            or perm(transpose) != (0, 2, 1, 3)
            or source[1] != 1
            or source[2] != target[2]
            or source[3] != target[1] * target[3]
        ):
            continue
        heads, width = target[1], target[3]
        split_name = f"modalix.head_pack_split.{serial}"
        split_outputs = [f"{transpose.output[0]}/head_{index}" for index in range(heads)]
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray([width] * heads, dtype=np.int64), name=split_name
            )
        )
        replacements[reshape.name] = [
            helper.make_node(
                "Split",
                [reshape.input[0], split_name],
                split_outputs,
                axis=3,
                name=f"{reshape.name}/modalix_head_split",
            ),
            helper.make_node(
                "Concat",
                split_outputs,
                list(transpose.output),
                axis=1,
                name=f"{transpose.name}/modalix_head_concat",
            ),
        ]
        skip.add(transpose.name)
        result["pack"] += 1
        serial += 1

    # [B,H,L,D] -> transpose [B,L,H,D] -> reshape [B,1,L,H*D]
    for transpose in model.graph.node:
        if transpose.op_type != "Transpose" or transpose.name in skip:
            continue
        children = uses.get(transpose.output[0], [])
        if len(children) != 1 or children[0].op_type != "Reshape":
            continue
        reshape = children[0]
        source = shapes.get(transpose.input[0])
        target = shapes.get(reshape.output[0])
        if (
            not source
            or not target
            or perm(transpose) != (0, 2, 1, 3)
            or target[1] != 1
            or source[2] != target[2]
            or source[1] * source[3] != target[3]
        ):
            continue
        heads = source[1]
        split_name = f"modalix.head_merge_split.{serial}"
        split_outputs = [f"{reshape.output[0]}/head_{index}" for index in range(heads)]
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray([1] * heads, dtype=np.int64), name=split_name
            )
        )
        replacements[transpose.name] = [
            helper.make_node(
                "Split",
                [transpose.input[0], split_name],
                split_outputs,
                axis=1,
                name=f"{transpose.name}/modalix_head_split",
            ),
            helper.make_node(
                "Concat",
                split_outputs,
                list(reshape.output),
                axis=3,
                name=f"{reshape.name}/modalix_head_concat",
            ),
        ]
        skip.add(reshape.name)
        result["merge"] += 1
        serial += 1

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in replacements:
            nodes.extend(replacements[node.name])
        elif node.name not in skip:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return result


def simplify_attention_masks(model: onnx.ModelProto) -> Counter:
    """Keep masks channel-first and remove masking dominated by the final mask.

    The lifted graph transposed the two public masks to channel-last even though
    every reduction spans all non-batch axes and the attention key mask needs
    the original [B,1,1,L] layout.  Attention probabilities were also multiplied
    by the query mask before the output projection and the projected result was
    multiplied by the same mask again.  The first multiplication cannot affect
    valid positions and is dominated by the final multiplication for invalid
    positions.

    Move the final multiplication across the channel-last-to-channel-first
    transpose.  This is an exact layout rewrite and lets every attention block
    consume the public latent mask directly.
    """

    latent_mask_cl = (
        "/vector_estimator/vector_field/main_blocks.3/Transpose_2_output_0"
    )
    text_mask_cl = (
        "/vector_estimator/vector_field/main_blocks.3/Transpose_3_output_0"
    )
    text_mask_cf = (
        "/vector_estimator/vector_field/main_blocks.3/attn/Transpose_1_output_0"
    )
    mask_transposes = {
        "/vector_estimator/vector_field/main_blocks.3/Transpose_2",
        "/vector_estimator/vector_field/main_blocks.3/Transpose_3",
        "/vector_estimator/vector_field/main_blocks.3/attn/Transpose_1",
    }

    uses = consumers(model)
    remove_names: set[str] = set(mask_transposes)
    insert_after: dict[str, onnx.NodeProto] = {}
    result = Counter()

    # The post-Softmax query mask is redundant because the projected attention
    # output is masked again before it is added to the residual stream.
    for node in model.graph.node:
        if node.op_type != "Mul":
            continue
        if "/attn/Where_1" not in node.name and "/attention/Where" not in node.name:
            continue
        replace_inputs(model, node.output[0], node.input[0], skip_name=node.name)
        remove_names.add(node.name)
        result["dominated_query_mul"] += 1

    # Transpose(x * mask_cl) == transpose(x) * mask_cf.  Put the final mask in
    # the graph's native channel-first layout so the shared mask transpose dies.
    for node in model.graph.node:
        if node.op_type != "Mul" or latent_mask_cl not in node.input:
            continue
        if not (node.name.endswith("/attn/Mul_14") or node.name.endswith("/attention/Mul")):
            continue
        children = uses.get(node.output[0], [])
        if len(children) != 1 or children[0].op_type != "Transpose":
            raise RuntimeError(f"unexpected final attention mask fanout: {node.name}")
        transpose = children[0]
        unmasked = next(value for value in node.input if value != latent_mask_cl)
        original_output = transpose.output[0]
        temporary_output = f"{original_output}/unmasked"
        transpose.input[0] = unmasked
        transpose.output[0] = temporary_output
        insert_after[transpose.name] = helper.make_node(
            "Mul",
            [temporary_output, "latent_mask"],
            [original_output],
            name=f"{node.name}/modalix_channel_first_mask",
        )
        remove_names.add(node.name)
        result["moved_final_mask"] += 1

    # The two reductions use axes [1,2,3], so their scalar result is invariant
    # to the mask layout.  The key-mask reciprocal already wants [B,1,1,L].
    replace_inputs(model, latent_mask_cl, "latent_mask")
    replace_inputs(model, text_mask_cl, "text_mask")
    replace_inputs(model, text_mask_cf, "text_mask")

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in remove_names:
            continue
        nodes.append(node)
        if node.name in insert_after:
            nodes.append(insert_after[node.name])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    result["removed_mask_transpose"] = len(mask_transposes)
    return result


def set_axis(node: onnx.NodeProto, axis: int) -> None:
    kept = [attr for attr in node.attribute if attr.name != "axis"]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute("axis", axis))


def rewrite_attention_like_llima(model: onnx.ModelProto) -> Counter:
    """Use the channel-first LLiMa attention layout and ONNX Einsum pattern.

    This follows ``llima/sima_lmm/model/language_cache_model.py``: Q/K/V use
    [B,D,H,L], scores use [B,K,H,Q], Softmax reduces K, and the second Einsum
    returns [B,D,H,Q].  Projection and output linears become 1x1 Conv2D, which
    means the convolutional residual stream never needs a layout transpose.
    """

    joint_blocks = (3, 9, 15, 21)
    style_blocks = (5, 11, 17, 23)
    by_name = {node.name: node for node in model.graph.node}
    producer = {out: node for node in model.graph.node for out in node.output}
    uses = consumers(model)
    initializer_proto = {value.name: value for value in model.graph.initializer}
    initializer_array = {
        name: numpy_helper.to_array(value) for name, value in initializer_proto.items()
    }
    replacements: dict[str, onnx.NodeProto] = {}
    remove_names: set[str] = set()
    result = Counter()

    def convert_linear(
        linear_prefix: str, source: str, output: str | None = None
    ) -> str:
        matmul = by_name[f"{linear_prefix}/linear/MatMul"]
        add = by_name[f"{linear_prefix}/linear/Add"]
        weight = initializer_array[matmul.input[1]]
        bias_names = [value for value in add.input if value in initializer_array]
        if weight.ndim != 2 or len(bias_names) != 1:
            raise RuntimeError(f"unexpected attention linear: {linear_prefix}")
        weight_name = f"{matmul.input[1]}/modalix_conv2d"
        model.graph.initializer.append(
            numpy_helper.from_array(weight.T[:, :, None, None], name=weight_name)
        )
        target = output or add.output[0]
        replacements[matmul.name] = helper.make_node(
            "Conv",
            [source, weight_name, bias_names[0]],
            [target],
            kernel_shape=[1, 1],
            name=f"{linear_prefix}/modalix_conv2d",
        )
        remove_names.add(add.name)
        result["linear_to_conv"] += 1
        return target

    def rewrite_pack(projected: str) -> str:
        split_nodes = [node for node in uses.get(projected, []) if node.op_type == "Split"]
        if len(split_nodes) != 1:
            raise RuntimeError(f"unexpected attention head pack: {projected}")
        split = split_nodes[0]
        concat_nodes = {
            node.name: node
            for value in split.output
            for node in uses.get(value, [])
            if node.op_type == "Concat"
        }
        if len(concat_nodes) != 1:
            raise RuntimeError(f"unexpected attention head concat: {split.name}")
        concat = next(iter(concat_nodes.values()))
        set_axis(split, 1)
        set_axis(concat, 2)
        result["head_pack_channel_first"] += 1
        return concat.output[0]

    def rewrite_merge(attention_output: str) -> str:
        split_nodes = [
            node for node in uses.get(attention_output, []) if node.op_type == "Split"
        ]
        if len(split_nodes) != 1:
            raise RuntimeError(f"unexpected attention head merge: {attention_output}")
        split = split_nodes[0]
        concat_nodes = {
            node.name: node
            for value in split.output
            for node in uses.get(value, [])
            if node.op_type == "Concat"
        }
        if len(concat_nodes) != 1:
            raise RuntimeError(f"unexpected attention merge concat: {split.name}")
        concat = next(iter(concat_nodes.values()))
        set_axis(split, 2)
        set_axis(concat, 1)
        result["head_merge_channel_first"] += 1
        return concat.output[0]

    def replace_attention_matmuls(
        prefix: str, query: str, key: str, value: str
    ) -> str:
        bmm1 = by_name[f"{prefix}/MatMul"]
        bmm2 = by_name[f"{prefix}/MatMul_1"]
        replacements[bmm1.name] = helper.make_node(
            "Einsum",
            [query, key],
            list(bmm1.output),
            equation="nchw,nchq->nqhw",
            name=f"{prefix}/bmm1_llima",
        )
        replacements[bmm2.name] = helper.make_node(
            "Einsum",
            [bmm2.input[0], value],
            list(bmm2.output),
            equation="nchw,nqhc->nqhw",
            name=f"{prefix}/bmm2_llima",
        )
        softmax = by_name[f"{prefix}/Softmax"]
        set_axis(softmax, 1)
        result["matmul_to_einsum"] += 2
        return bmm2.output[0]

    # Public context tensors use NCHW too.  The host transposes the two compact
    # style arrays once while constructing a request; no graph layout node is
    # needed.  The text mask follows LLiMa's [B,K,1,Q-broadcast] score layout.
    input_shapes = {
        "style_ttl": (1, 256, 1, 50),
        "style_key": (1, 256, 1, 50),
        "text_mask": (1, 192, 1, 1),
    }
    for value in model.graph.input:
        if value.name not in input_shapes:
            continue
        for dim, size in zip(value.type.tensor_type.shape.dim, input_shapes[value.name]):
            dim.ClearField("dim_param")
            dim.dim_value = size

    # Generate rotary angles directly as [B,D/2,1,L].
    increments_name = "vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.increments"
    theta_name = "vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.theta"
    increments = initializer_array[increments_name].transpose(0, 1, 3, 2)
    theta = initializer_array[theta_name].transpose(0, 3, 1, 2)
    for name, array in ((increments_name, increments), (theta_name, theta)):
        old = initializer_proto[name]
        index = next(i for i, value in enumerate(model.graph.initializer) if value.name == name)
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name=name))
    for slice_name in (
        "/vector_estimator/vector_field/main_blocks.3/attn/Slice",
        "/vector_estimator/vector_field/main_blocks.3/attn/Slice_3",
    ):
        slice_node = by_name[slice_name]
        axes_name = slice_node.input[3]
        index = next(
            i for i, value in enumerate(model.graph.initializer) if value.name == axes_name
        )
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(np.asarray([3], dtype=np.int64), name=axes_name)
        )

    remove_names.add("/vector_estimator/vector_field/main_blocks.3/Transpose_1")

    for block in joint_blocks:
        parent = f"/vector_estimator/vector_field/main_blocks.{block}"
        prefix = f"{parent}/attn"
        input_transpose = by_name[f"{parent}/Transpose"]
        output_transpose_name = (
            f"{parent}/Transpose_4" if block == 3 else f"{parent}/Transpose_1"
        )
        output_transpose = by_name[output_transpose_name]
        remove_names.update((input_transpose.name, output_transpose.name))

        query_projected = convert_linear(
            f"{prefix}/W_query", input_transpose.input[0]
        )
        key_projected = convert_linear(f"{prefix}/W_key", "text_emb")
        value_projected = convert_linear(f"{prefix}/W_value", "text_emb")
        query = rewrite_pack(query_projected)
        key = rewrite_pack(key_projected)
        value = rewrite_pack(value_projected)

        # RoPE now splits and concatenates the D axis rather than the last axis.
        for suffix in ("Slice_1", "Slice_2", "Slice_4", "Slice_5"):
            slice_node = by_name[f"{prefix}/{suffix}"]
            axes_name = slice_node.input[3]
            index = next(
                i
                for i, initializer in enumerate(model.graph.initializer)
                if initializer.name == axes_name
            )
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=axes_name)
            )
        set_axis(by_name[f"{prefix}/Concat_3"], 1)
        set_axis(by_name[f"{prefix}/Concat_4"], 1)
        query = by_name[f"{prefix}/Concat_3"].output[0]
        key = by_name[f"{prefix}/Concat_4"].output[0]
        key_transpose = by_name[f"{prefix}/Transpose"]
        remove_names.add(key_transpose.name)

        attention_output = replace_attention_matmuls(prefix, query, key, value)
        merged = rewrite_merge(attention_output)
        mask = by_name[f"{prefix}/Mul_14/modalix_channel_first_mask"]
        unmasked = next(value for value in mask.input if value != "latent_mask")
        convert_linear(f"{prefix}/out_fc", merged, unmasked)

    for block in style_blocks:
        parent = f"/vector_estimator/vector_field/main_blocks.{block}"
        prefix = f"{parent}/attention"
        input_transpose = by_name[f"{parent}/Transpose"]
        output_transpose = by_name[f"{parent}/Transpose_1"]
        remove_names.update((input_transpose.name, output_transpose.name))

        query_projected = convert_linear(
            f"{prefix}/W_query", input_transpose.input[0]
        )
        key_projected = convert_linear(f"{prefix}/W_key", "style_key")
        value_projected = convert_linear(f"{prefix}/W_value", "style_ttl")
        query = rewrite_pack(query_projected)
        key = rewrite_pack(key_projected)
        value = rewrite_pack(value_projected)
        key_transpose = by_name[f"{prefix}/Transpose"]
        tanh = by_name[f"{prefix}/tanh/Tanh"]
        tanh.input[0] = key
        remove_names.add(key_transpose.name)
        key = tanh.output[0]

        attention_output = replace_attention_matmuls(prefix, query, key, value)
        merged = rewrite_merge(attention_output)
        mask = by_name[f"{prefix}/Mul/modalix_channel_first_mask"]
        unmasked = next(value for value in mask.input if value != "latent_mask")
        convert_linear(f"{prefix}/out_fc", merged, unmasked)

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in replacements:
            nodes.append(replacements[node.name])
        elif node.name not in remove_names:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    result["removed_transposes"] = sum(
        by_name[name].op_type == "Transpose" for name in remove_names
    )
    return result


def rewrite_time_encoder_channel_first(model: onnx.ModelProto) -> Counter:
    """Keep the sinusoidal time MLP in NCHW and replace its FCs with Conv2D."""

    frequency_name = (
        "/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0"
    )
    index = next(
        i for i, value in enumerate(model.graph.initializer) if value.name == frequency_name
    )
    frequencies = numpy_helper.to_array(model.graph.initializer[index]).reshape(1, 32, 1, 1)
    model.graph.initializer[index].CopyFrom(
        numpy_helper.from_array(frequencies, name=frequency_name)
    )
    concat_name = "/vector_estimator/vector_field/time_encoder/sinusoidal/Concat"
    concat = next(node for node in model.graph.node if node.name == concat_name)
    set_axis(concat, 1)

    by_name = {node.name: node for node in model.graph.node}
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    replacements: dict[str, onnx.NodeProto] = {}
    remove_names: set[str] = set()
    for layer in (0, 2):
        prefix = f"/vector_estimator/vector_field/time_encoder/mlp/mlp.{layer}/linear/Gemm"
        matmul = by_name[f"{prefix}/modalix_matmul"]
        add = by_name[prefix]
        weight = initializers[matmul.input[1]]
        bias_name = next(value for value in add.input if value in initializers)
        weight_name = f"{matmul.input[1]}/modalix_conv2d"
        model.graph.initializer.append(
            numpy_helper.from_array(weight.T[:, :, None, None], name=weight_name)
        )
        replacements[matmul.name] = helper.make_node(
            "Conv",
            [matmul.input[0], weight_name, bias_name],
            list(add.output),
            kernel_shape=[1, 1],
            name=f"{prefix}/modalix_conv2d",
        )
        remove_names.add(add.name)

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in replacements:
            nodes.append(replacements[node.name])
        elif node.name not in remove_names:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return Counter(linear_to_conv=2, removed_transposes=1)


def make_edge_padding_mask_aware(model: onnx.ModelProto) -> Counter:
    """Make fixed-width edge padding behave like the natural latent boundary.

    The released dynamic graph pads each ConvNeXt activation at its physical
    sequence endpoint.  With a fixed L=192 tensor, zero-masked tail positions
    move that endpoint and change valid samples.  Before each edge Pad, replace
    the invalid tail by the activation at the final valid position.  The valid
    prefix is untouched and the existing edge Pad then sees exactly the same
    boundary as a natural-length invocation.
    """

    edge_pads = [
        node
        for node in model.graph.node
        if node.op_type == "Pad"
        and any(attr.name == "mode" and attr.s == b"edge" for attr in node.attribute)
    ]
    if not edge_pads:
        return Counter()

    starts_name = "modalix.latent_mask_next.starts"
    ends_name = "modalix.latent_mask_next.ends"
    axes_name = "modalix.latent_mask_next.axes"
    zero_name = "modalix.latent_mask_next.zero"
    one_name = "modalix.latent_mask_next.one"
    reduce_axes_name = "modalix.edge_fill.reduce_axes"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=starts_name),
            numpy_helper.from_array(np.asarray([192], dtype=np.int64), name=ends_name),
            numpy_helper.from_array(np.asarray([3], dtype=np.int64), name=axes_name),
            numpy_helper.from_array(
                np.zeros((1, 1, 1, 1), dtype=np.float32), name=zero_name
            ),
            numpy_helper.from_array(
                np.ones((1, 1, 1, 1), dtype=np.float32), name=one_name
            ),
            numpy_helper.from_array(
                np.asarray([3], dtype=np.int64), name=reduce_axes_name
            ),
        ]
    )
    next_slice = "modalix.latent_mask_next.slice"
    next_mask = "modalix.latent_mask_next"
    boundary = "modalix.latent_mask_boundary"
    invalid = "modalix.latent_mask_invalid"
    prefix_nodes = [
        helper.make_node(
            "Slice",
            ["latent_mask", starts_name, ends_name, axes_name],
            [next_slice],
            name="/modalix/latent_mask_next/Slice",
        ),
        helper.make_node(
            "Concat",
            [next_slice, zero_name],
            [next_mask],
            axis=3,
            name="/modalix/latent_mask_next/Concat",
        ),
        helper.make_node(
            "Sub",
            ["latent_mask", next_mask],
            [boundary],
            name="/modalix/latent_mask_boundary",
        ),
        helper.make_node(
            "Sub",
            [one_name, "latent_mask"],
            [invalid],
            name="/modalix/latent_mask_invalid",
        ),
    ]

    before_pad: dict[str, list[onnx.NodeProto]] = {}
    for serial, pad in enumerate(edge_pads):
        source = pad.input[0]
        weighted = f"{pad.output[0]}/modalix_boundary_weighted"
        last = f"{pad.output[0]}/modalix_last_valid"
        delta = f"{pad.output[0]}/modalix_tail_delta"
        tail = f"{pad.output[0]}/modalix_tail"
        filled = f"{pad.output[0]}/modalix_edge_filled"
        before_pad[pad.name] = [
            helper.make_node(
                "Mul",
                [source, boundary],
                [weighted],
                name=f"{pad.name}/modalix_boundary_weight",
            ),
            helper.make_node(
                "ReduceSum",
                [weighted, reduce_axes_name],
                [last],
                keepdims=1,
                name=f"{pad.name}/modalix_last_valid",
            ),
            helper.make_node(
                "Sub",
                [last, source],
                [delta],
                name=f"{pad.name}/modalix_tail_delta",
            ),
            helper.make_node(
                "Mul",
                [invalid, delta],
                [tail],
                name=f"{pad.name}/modalix_tail",
            ),
            helper.make_node(
                "Add",
                [source, tail],
                [filled],
                name=f"{pad.name}/modalix_edge_fill",
            ),
        ]
        pad.input[0] = filled

    nodes: list[onnx.NodeProto] = list(prefix_nodes)
    for node in model.graph.node:
        nodes.extend(before_pad.get(node.name, []))
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return Counter(edge_pads=len(edge_pads), added_nodes=len(prefix_nodes) + 5 * len(edge_pads))


def replace_edge_pads_with_constant_padding(model: onnx.ModelProto) -> Counter:
    """Express edge padding with zero Pad plus broadcast boundary corrections.

    MLA supports only constant-zero padding.  For fixed width 192, zero-pad the
    activation, slice its two boundary columns, broadcast each against a mask
    covering its padding region, then add those corrections.  This is exactly
    ``Pad(mode="edge")`` while keeping every operation on supported MLA paths.
    """

    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    starts_left = "modalix.edge_pad.left.starts"
    ends_left = "modalix.edge_pad.left.ends"
    starts_right = "modalix.edge_pad.right.starts"
    ends_right = "modalix.edge_pad.right.ends"
    axes = "modalix.edge_pad.axes"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=starts_left),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=ends_left),
            numpy_helper.from_array(
                np.asarray([191], dtype=np.int64), name=starts_right
            ),
            numpy_helper.from_array(
                np.asarray([192], dtype=np.int64), name=ends_right
            ),
            numpy_helper.from_array(np.asarray([3], dtype=np.int64), name=axes),
        ]
    )

    mask_names: dict[tuple[int, int], tuple[str, str]] = {}
    nodes: list[onnx.NodeProto] = []
    result = Counter()
    for node in model.graph.node:
        if node.op_type != "Pad":
            nodes.append(node)
            continue
        mode = next(
            (attr.s for attr in node.attribute if attr.name == "mode"), b"constant"
        )
        pads = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if mode != b"edge" or pads is None:
            nodes.append(node)
            continue
        pads = np.asarray(pads).reshape(-1)
        if pads.size != 8 or np.any(pads[[0, 1, 2, 4, 5, 6]] != 0):
            raise RuntimeError(f"unsupported edge-pad geometry: {node.name}: {pads}")
        left_width, right_width = int(pads[3]), int(pads[7])
        if left_width < 0 or right_width < 0:
            raise RuntimeError(f"negative edge padding: {node.name}: {pads}")

        geometry = (left_width, right_width)
        if geometry not in mask_names:
            output_width = 192 + left_width + right_width
            left_mask = np.zeros((1, 1, 1, output_width), dtype=np.float32)
            right_mask = np.zeros((1, 1, 1, output_width), dtype=np.float32)
            left_mask[..., :left_width] = 1.0
            if right_width:
                right_mask[..., -right_width:] = 1.0
            left_mask_name = f"modalix.edge_pad.mask.{left_width}.{right_width}.left"
            right_mask_name = f"modalix.edge_pad.mask.{left_width}.{right_width}.right"
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(left_mask, name=left_mask_name),
                    numpy_helper.from_array(right_mask, name=right_mask_name),
                ]
            )
            mask_names[geometry] = (left_mask_name, right_mask_name)
        left_mask_name, right_mask_name = mask_names[geometry]

        left = f"{node.output[0]}/modalix_left_edge"
        right = f"{node.output[0]}/modalix_right_edge"
        zero_padded = f"{node.output[0]}/modalix_zero_padded"
        left_padding = f"{node.output[0]}/modalix_left_padding"
        right_padding = f"{node.output[0]}/modalix_right_padding"
        with_left = f"{node.output[0]}/modalix_with_left_edge"
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [node.input[0], starts_left, ends_left, axes],
                    [left],
                    name=f"{node.name}/modalix_left_edge",
                ),
                helper.make_node(
                    "Slice",
                    [node.input[0], starts_right, ends_right, axes],
                    [right],
                    name=f"{node.name}/modalix_right_edge",
                ),
                helper.make_node(
                    "Pad",
                    [node.input[0], node.input[1]],
                    [zero_padded],
                    mode="constant",
                    name=f"{node.name}/modalix_zero_pad",
                ),
                helper.make_node(
                    "Mul",
                    [left, left_mask_name],
                    [left_padding],
                    name=f"{node.name}/modalix_left_padding",
                ),
                helper.make_node(
                    "Mul",
                    [right, right_mask_name],
                    [right_padding],
                    name=f"{node.name}/modalix_right_padding",
                ),
                helper.make_node(
                    "Add",
                    [zero_padded, left_padding],
                    [with_left],
                    name=f"{node.name}/modalix_add_left_edge",
                ),
                helper.make_node(
                    "Add",
                    [with_left, right_padding],
                    list(node.output),
                    name=f"{node.name}/modalix_add_right_edge",
                ),
            ]
        )
        result["edge_pad_to_zero_pad_and_boundary_add"] += 1

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return result


def specialize_singleton_reductions(model: onnx.ModelProto) -> Counter:
    """Remove singleton channel/height axes from reduction axis lists.

    A reduction over ``[1, 1, 1, 192]`` axes ``[1, 2, 3]`` equals a reduction
    over axis ``[3]``.  Stating only the non-singleton spatial axis prevents the
    compiler from treating this mask-length calculation as a channel reduction.
    """

    # The optimizer deliberately changes several physical layouts before the
    # final value-info refresh, so use the current declared shapes here rather
    # than running shape inference on stale intermediate annotations.
    shapes = shape_map(model)
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    result = Counter()
    serial = 0
    for node in model.graph.node:
        if node.op_type not in {"ReduceSum", "ReduceMean"} or len(node.input) < 2:
            continue
        shape = shapes.get(node.input[0])
        axes = initializers.get(node.input[1])
        if not shape or axes is None:
            continue
        normalized = [
            int(axis if axis >= 0 else axis + len(shape))
            for axis in np.asarray(axes).reshape(-1)
        ]
        reduced = [axis for axis in normalized if shape[axis] != 1]
        if not reduced or reduced == normalized:
            continue
        axes_name = f"modalix.non_singleton_reduce_axes.{serial}"
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(reduced, dtype=np.int64), name=axes_name)
        )
        node.input[1] = axes_name
        result[f"{node.op_type}_singleton_axes_removed"] += len(normalized) - len(reduced)
        serial += 1
    return result


def fuse_time_projections(model: onnx.ModelProto) -> Counter:
    """Fuse four identical-layout time projections into one channel-first Conv.

    Each stage independently computes ``time_embedding @ weight + bias`` in
    channel-last layout and then transposes the scalar spatial tensor to
    channel-first.  A single input transpose followed by a fused 1x1 Conv and a
    channel Split computes the same four projections without four copies of the
    layout conversion.
    """

    transpose_names = [
        "/vector_estimator/vector_field/main_blocks.1/Transpose_1",
        "/vector_estimator/vector_field/main_blocks.7/Transpose",
        "/vector_estimator/vector_field/main_blocks.13/Transpose",
        "/vector_estimator/vector_field/main_blocks.19/Transpose",
    ]
    by_name = {node.name: node for node in model.graph.node}
    producer = {out: node for node in model.graph.node for out in node.output}
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    chains: list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]] = []
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    source = ""
    outputs: list[str] = []

    for name in transpose_names:
        transpose = by_name.get(name)
        if transpose is None or transpose.op_type != "Transpose":
            raise RuntimeError(f"missing time projection transpose: {name}")
        add = producer.get(transpose.input[0])
        if add is None or add.op_type != "Add":
            raise RuntimeError(f"unexpected time projection output: {name}")
        matmul_inputs = [value for value in add.input if value in producer]
        if len(matmul_inputs) != 1:
            raise RuntimeError(f"unexpected time projection Add: {add.name}")
        matmul = producer[matmul_inputs[0]]
        if matmul.op_type != "MatMul" or matmul.input[1] not in initializers:
            raise RuntimeError(f"unexpected time projection MatMul: {matmul.name}")
        bias_names = [value for value in add.input if value in initializers]
        if len(bias_names) != 1:
            raise RuntimeError(f"unexpected time projection bias: {add.name}")
        if source and source != matmul.input[0]:
            raise RuntimeError("time projections do not share one embedding")
        source = matmul.input[0]
        weights.append(initializers[matmul.input[1]])
        biases.append(initializers[bias_names[0]])
        outputs.append(transpose.output[0])
        chains.append((matmul, add, transpose))

    if any(weight.shape != weights[0].shape for weight in weights):
        raise RuntimeError("time projection weights have different shapes")
    if any(bias.shape != biases[0].shape for bias in biases):
        raise RuntimeError("time projection biases have different shapes")
    channels_in, channels_out = weights[0].shape
    fused_weight = np.concatenate(weights, axis=1).T[:, :, None, None]
    fused_bias = np.concatenate(biases)
    weight_name = "modalix.time_projections.weight"
    bias_name = "modalix.time_projections.bias"
    split_name = "modalix.time_projections.split"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(fused_weight, name=weight_name),
            numpy_helper.from_array(fused_bias, name=bias_name),
            numpy_helper.from_array(
                np.asarray([channels_out] * len(outputs), dtype=np.int64),
                name=split_name,
            ),
        ]
    )

    fused_output = f"{source}/modalix_fused_time_projections"
    replacement = [
        helper.make_node(
            "Conv",
            [source, weight_name, bias_name],
            [fused_output],
            kernel_shape=[1, 1],
            name="/modalix/time_projections/Conv",
        ),
        helper.make_node(
            "Split",
            [fused_output, split_name],
            outputs,
            axis=1,
            name="/modalix/time_projections/Split",
        ),
    ]
    remove_names = {node.name for chain in chains for node in chain}
    first_name = chains[0][0].name
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name == first_name:
            nodes.extend(replacement)
        if node.name not in remove_names:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return Counter(
        fused_projections=len(outputs),
        removed_nodes=len(remove_names),
        replacement_nodes=len(replacement),
        removed_transposes=len(outputs),
        input_channels=channels_in,
        output_channels=len(outputs) * channels_out,
    )


def prune_initializers(model: onnx.ModelProto) -> int:
    used = {value for node in model.graph.node for value in node.input}
    old = len(model.graph.initializer)
    kept = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return old - len(kept)


def optimize(source: Path, output: Path) -> dict[str, object]:
    model = onnx.load(source)
    source_node_count = len(model.graph.node)
    before = Counter(node.op_type for node in model.graph.node)
    removed_identities = remove_identities(model)
    removed_noop_reshapes = remove_noop_reshapes(model)
    singleton = remove_singleton_reshape_transpose_fanouts(model)
    heads = replace_head_pack_and_merge(model)
    masks = simplify_attention_masks(model)
    attention = rewrite_attention_like_llima(model)
    time_encoder = rewrite_time_encoder_channel_first(model)
    time_projections = fuse_time_projections(model)
    edge_padding = make_edge_padding_mask_aware(model)
    edge_pad_decomposition = replace_edge_pads_with_constant_padding(model)
    singleton_reductions = specialize_singleton_reductions(model)
    removed_initializers = prune_initializers(model)
    del model.graph.value_info[:]
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    checker.check_model(inferred, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)
    after = Counter(node.op_type for node in inferred.graph.node)
    forbidden = {
        op: after[op]
        for op in (
            "Identity",
            "Reshape",
            "Gather",
            "Squeeze",
            "Unsqueeze",
            "Equal",
            "Where",
        )
    }
    if any(forbidden.values()):
        raise RuntimeError(f"wiring cleanup incomplete: {forbidden}")
    non_norm_transposes = [
        node.name
        for node in inferred.graph.node
        if node.op_type == "Transpose" and "/norm/" not in node.name
    ]
    if non_norm_transposes:
        raise RuntimeError(f"non-LayerNorm transposes remain: {non_norm_transposes}")
    fp_types = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16}
    rank_violations: list[dict[str, object]] = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(
        inferred.graph.output
    ):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type not in fp_types:
            continue
        dimensions = [dim.dim_value for dim in tensor_type.shape.dim]
        if len(dimensions) != 4 or dimensions[0] != 1:
            rank_violations.append({"name": value.name, "shape": dimensions})
    if rank_violations:
        raise RuntimeError(f"rank-4 activation violations: {rank_violations[:10]}")
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "before_nodes": source_node_count,
        "after_nodes": len(inferred.graph.node),
        "removed_identities": removed_identities,
        "removed_noop_reshapes": removed_noop_reshapes,
        "removed_singleton_fanout": dict(singleton),
        "head_rewrites": dict(heads),
        "mask_rewrites": dict(masks),
        "attention_rewrite": dict(attention),
        "time_encoder_rewrite": dict(time_encoder),
        "time_projection_rewrite": dict(time_projections),
        "edge_padding_rewrite": dict(edge_padding),
        "edge_pad_decomposition": dict(edge_pad_decomposition),
        "singleton_reduction_rewrite": dict(singleton_reductions),
        "removed_initializers": removed_initializers,
        "forbidden_ops": forbidden,
        "non_layernorm_transposes": non_norm_transposes,
        "rank4_leading_batch_violations": rank_violations,
        "before_ops": dict(sorted(before.items())),
        "after_ops": dict(sorted(after.items())),
    }


# Timestep and RoPE externalization.


TIME_PAIR = "/vector_estimator/vector_field/time_encoder/sinusoidal/Concat_output_0"
ROTARY_VALUES = (
    "/vector_estimator/vector_field/main_blocks.3/attn/Sin_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Cos_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Sin_1_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Cos_1_output_0",
)


def value_info(model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value.name == name:
            return copy.deepcopy(value)
    raise KeyError(f"missing value info for {name}")


def intermediate_session(model: onnx.ModelProto, names: list[str]) -> ort.InferenceSession:
    exposed = copy.deepcopy(model)
    existing = {value.name for value in exposed.graph.output}
    for name in names:
        if name not in existing:
            exposed.graph.output.append(value_info(exposed, name))
    with tempfile.NamedTemporaryFile(suffix=".onnx") as temporary:
        onnx.save(exposed, temporary.name)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        return ort.InferenceSession(
            temporary.name,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )


def prune_dead_graph(model: onnx.ModelProto) -> tuple[int, int]:
    needed = {value.name for value in model.graph.output}
    kept_reversed: list[onnx.NodeProto] = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept_reversed.append(node)
            needed.update(value for value in node.input if value)
    kept = list(reversed(kept_reversed))
    removed_nodes = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    used = {value for node in model.graph.node for value in node.input}
    initializers = [value for value in model.graph.initializer if value.name in used]
    removed_initializers = len(model.graph.initializer) - len(initializers)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return removed_nodes, removed_initializers


def ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def externalize(argv: list[str] | None = None, *, emit: bool = True) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-wrapper", type=Path, required=True)
    parser.add_argument(
        "--reference-case",
        type=Path,
        required=True,
        nargs="+",
        help=(
            "One or more padded reference NPZs. The first generates the tables; "
            "all cases validate the rewritten graph."
        ),
    )
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)

    model = shape_inference.infer_shapes(onnx.load(args.input), strict_mode=True)
    validation_cases = {
        path.stem: load_case(path) for path in args.reference_case
    }
    arrays = next(iter(validation_cases.values()))
    constants = source_constants(args.source_wrapper)
    probe = intermediate_session(model, [TIME_PAIR, *ROTARY_VALUES])

    time_rows = []
    base_feeds = None
    for step in range(args.steps):
        feeds = pre_external_inputs(
            arrays,
            constants,
            arrays["noisy_latent_padded"],
            step,
            args.steps,
            True,
        )
        time_rows.append(np.asarray(probe.run([TIME_PAIR], feeds)[0], dtype=np.float32))
        if step == 0:
            base_feeds = feeds
    assert base_feeds is not None
    time_table = np.stack(time_rows, axis=0)
    expected_time_shape = (args.steps, *TIME_INPUT_SHAPE)
    if time_table.shape != expected_time_shape:
        raise RuntimeError(f"time table shape {time_table.shape} != {expected_time_shape}")

    # Both rotary branches use the same increments and theta.  Materialize the
    # original ONNX values for every legal effective length instead of
    # duplicating the formula here, and prove the latent/text branches agree.
    rope_rows = []
    shared_branch_max_abs = 0.0
    for length in range(1, MAX_SEQUENCE_LENGTH + 1):
        feeds = {name: value.copy() for name, value in base_feeds.items()}
        feeds["latent_mask"].fill(0.0)
        feeds["latent_mask"][..., :length] = 1.0
        feeds["text_mask"].fill(0.0)
        feeds["text_mask"][:, :length, ...] = 1.0
        latent_sin, latent_cos, text_sin, text_cos = [
            np.asarray(value, dtype=np.float32)
            for value in probe.run(list(ROTARY_VALUES), feeds)
        ]
        shared_branch_max_abs = max(
            shared_branch_max_abs,
            float(np.max(np.abs(latent_sin - text_sin))),
            float(np.max(np.abs(latent_cos - text_cos))),
        )
        if not (
            np.array_equal(latent_sin, text_sin)
            and np.array_equal(latent_cos, text_cos)
        ):
            raise RuntimeError(
                f"latent/text RoPE branches differ at effective length {length}"
            )
        rope_rows.append(np.concatenate([latent_sin, latent_cos], axis=1))
    rope_table = np.stack(rope_rows, axis=0)
    if rope_table.shape != ROPE_BANK_SHAPE:
        raise RuntimeError(f"RoPE bank shape {rope_table.shape} != {ROPE_BANK_SHAPE}")

    modified = copy.deepcopy(model)
    consumers = 0
    for node in modified.graph.node:
        for index, name in enumerate(node.input):
            if name == TIME_PAIR:
                node.input[index] = TIME_INPUT
                consumers += 1
    if consumers != 1:
        raise RuntimeError(f"expected one time-pair consumer, found {consumers}")

    inputs = [
        value
        for value in modified.graph.input
        if value.name not in {"current_step", "total_step"}
    ]
    inputs.append(
        helper.make_tensor_value_info(TIME_INPUT, TensorProto.FLOAT, TIME_INPUT_SHAPE)
    )
    inputs.append(
        helper.make_tensor_value_info(ROPE_INPUT, TensorProto.FLOAT, ROPE_INPUT_SHAPE)
    )
    del modified.graph.input[:]
    modified.graph.input.extend(inputs)

    rotary_names = set(ROTARY_VALUES)
    nodes = [
        node
        for node in modified.graph.node
        if not any(output in rotary_names for output in node.output)
    ]
    del modified.graph.node[:]
    split_sizes_name = "modalix.rope_tables.split_sizes"
    modified.graph.node.extend(
        [
            helper.make_node(
                "Split",
                [ROPE_INPUT, split_sizes_name],
                list(ROTARY_VALUES),
                name="/modalix/rope_tables/Split",
                axis=1,
            ),
            *nodes,
        ]
    )
    modified.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([32, 32, 32, 32], dtype=np.int64), name=split_sizes_name
        )
    )

    removed_nodes, removed_initializers = prune_dead_graph(modified)
    del modified.graph.value_info[:]
    modified = shape_inference.infer_shapes(modified, strict_mode=True, data_prop=True)
    checker.check_model(modified, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(modified, args.output)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.table,
        **{
            ROPE_TABLE_KEY: rope_table,
            **constants,
        },
    )

    original_runtime = ort_session(args.input)
    modified_runtime = ort_session(args.output)
    comparisons = []
    for case_name, case_arrays in validation_cases.items():
        for step in range(args.steps):
            for conditional in (True, False):
                feeds = pre_external_inputs(
                    case_arrays,
                    constants,
                    case_arrays["noisy_latent_padded"],
                    step,
                    args.steps,
                    conditional,
                )
                reference = original_runtime.run(None, feeds)[0]
                candidate_feeds = {
                    name: value
                    for name, value in feeds.items()
                    if name not in {"current_step", "total_step"}
                }
                candidate_feeds[TIME_INPUT] = time_table[step]
                candidate_feeds[ROPE_INPUT] = pack_rope_input(
                    rope_table,
                    feeds["latent_mask"],
                    feeds["text_mask"],
                )
                candidate = modified_runtime.run(None, candidate_feeds)[0]
                comparisons.append(
                    {
                        "case": case_name,
                        "latent_length": int(feeds["latent_mask"].sum()),
                        "text_length": int(feeds["text_mask"].sum()),
                        "step": step,
                        "branch": (
                            "conditional" if conditional else "unconditional"
                        ),
                        **metrics(reference, candidate),
                    }
                )

    report = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "table": str(args.table),
        "table_sha256": sha256(args.table),
        "time_table_shape": list(time_table.shape),
        "rope_table_shape": list(rope_table.shape),
        "rope_input_shape": list(ROPE_INPUT_SHAPE),
        "rope_pack_order": list(ROTARY_VALUES),
        "rope_shared_branch_max_abs": shared_branch_max_abs,
        "runtime_constants": {
            name: list(value.shape) for name, value in constants.items()
        },
        "validation_cases": list(validation_cases),
        "removed_nodes": removed_nodes,
        "removed_initializers": removed_initializers,
        "input_names": [value.name for value in modified.graph.input],
        "node_count": len(modified.graph.node),
        "op_counts": dict(
            sorted(
                {
                    op: sum(node.op_type == op for node in modified.graph.node)
                    for op in {node.op_type for node in modified.graph.node}
                }.items()
            )
        ),
        "validation": {
            "comparisons": comparisons,
            "max_abs": max(value["max_abs"] for value in comparisons),
            "max_relative_l2": max(value["relative_l2"] for value in comparisons),
            "min_cosine": min(value["cosine"] for value in comparisons),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if emit:
        print(json.dumps(report, indent=2))
    return report
