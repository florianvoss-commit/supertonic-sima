#!/usr/bin/env python3
"""Freeze Supertonic 3 vector-estimator dimensions and validate the ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx

from model_contract import (
    HF_REPOSITORY,
    HF_REVISION,
    INPUT_SHAPES,
    OUTPUT_SHAPES,
    SOURCE_SHA256,
    SUPERTONIC_VERSION,
    SYMBOLIC_DIMENSIONS,
    serializable_contract,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--simplified-output", type=Path)
    args = parser.parse_args()

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
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
