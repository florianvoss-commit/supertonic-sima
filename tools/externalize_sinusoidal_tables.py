#!/usr/bin/env python3
"""Externalize timestep and length-dependent RoPE sin/cos inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vector_field_utils import (  # noqa: E402
    load_case,
    metrics,
    pre_external_inputs,
    source_constants,
)
from sinusoidal_inputs import (  # noqa: E402
    MAX_SEQUENCE_LENGTH,
    ROPE_BANK_SHAPE,
    ROPE_INPUT,
    ROPE_INPUT_SHAPE,
    ROPE_TABLE_KEY,
    TIME_INPUT,
    TIME_INPUT_SHAPE,
    TIME_TABLE_KEY,
    pack_rope_input,
)


TIME_PAIR = "/vector_estimator/vector_field/time_encoder/sinusoidal/Concat_output_0"
ROTARY_VALUES = (
    "/vector_estimator/vector_field/main_blocks.3/attn/Sin_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Cos_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Sin_1_output_0",
    "/vector_estimator/vector_field/main_blocks.3/attn/Cos_1_output_0",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> int:
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
    args = parser.parse_args()

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
            TIME_TABLE_KEY: time_table,
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
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
