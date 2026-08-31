#!/usr/bin/env python3
"""Remove redundant rank-4 wiring from the Supertonic vector-field graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "removed_initializers": removed_initializers,
        "forbidden_ops": forbidden,
        "non_layernorm_transposes": non_norm_transposes,
        "rank4_leading_batch_violations": rank_violations,
        "before_ops": dict(sorted(before.items())),
        "after_ops": dict(sorted(after.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = optimize(args.source, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
