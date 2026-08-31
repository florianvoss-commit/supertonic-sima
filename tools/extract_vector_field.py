#!/usr/bin/env python3
"""Extract the batch-1 vector-field core from Supertonic's CFG wrapper.

The released vector_estimator ONNX builds a two-element batch internally for
classifier-free guidance and returns the result of the Euler update.  Modalix
needs a real leading batch of one.  This tool removes the wrapper and exposes a
single conditional vector-field evaluation.  The caller performs the
conditional and unconditional evaluations and combines them as ``4*c - 3*u``.

This is deliberately a semantics-preserving extraction only.  Rank lifting to
the final all-4D contract is a separate transformation and validation step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper, shape_inference


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = extract(args.source, args.output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
