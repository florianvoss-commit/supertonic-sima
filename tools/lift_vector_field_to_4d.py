#!/usr/bin/env python3
"""Lift the fixed Supertonic batch-1 vector field to rank-4 activations.

Data tensors use one of three physical layouts:

* channel first: ``[B, C, 1, L]``
* channel last:  ``[B, 1, L, C]``
* attention:     ``[B, H, L, D]``

The source graph is already fixed to B=1, text=192, latent=192.  Shape/index
bookkeeping tensors are intentionally left at their natural rank; every public
tensor and every floating-point data activation is lifted to rank four.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    Pre-softmax key masking becomes ``scores + (1 - reciprocal(mask))``.  For a
    binary mask this is exactly 0 for valid keys and -inf for invalid keys.
    Post-softmax query masking becomes a direct multiply by the binary mask.
    """

    producer = {out: node for node in model.graph.node for out in node.output}
    initializers = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    one_name = "modalix.mask_one"
    if one_name not in initializers:
        make_initializer(model, one_name, np.asarray(1.0, dtype=np.float32))

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
            reciprocal = f"{node.output[0]}/modalix_mask_reciprocal"
            bias = f"{node.output[0]}/modalix_mask_bias"
            new_nodes.extend(
                [
                    helper.make_node(
                        "Reciprocal",
                        [mask],
                        [reciprocal],
                        name=f"{node.name}/modalix_mask_reciprocal",
                    ),
                    helper.make_node(
                        "Sub",
                        [one_name, reciprocal],
                        [bias],
                        name=f"{node.name}/modalix_mask_bias",
                    ),
                    helper.make_node("Add", [data, bias], list(node.output), name=node.name),
                ]
            )
            rewritten["Where->ReciprocalSubAdd"] += 1
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = lift(args.source, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
