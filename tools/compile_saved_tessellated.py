#!/usr/bin/env python3
"""Compile a saved quantized AFE model with direct MLA I/O tessellation."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
from afe.apis.model import Model
from afe.ir.node import node_is_tuple


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-directory", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--compiler-debug-directory", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Effective leading batch dimension exposed by the compiled MPK.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    model = Model.load(
        model_name=args.model_name,
        network_directory=str(args.network_directory),
        include_unquantized_net=False,
        log_level=logging.INFO,
    )
    segments = list(model._net.sub_graph_names)
    if segments != ["MLA_0"]:
        raise RuntimeError(f"expected one MLA segment, found {segments}")

    mla_node = model._net.nodes["MLA_0"]
    input_parameters = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0),
        enable_mla=True,
        dram_layout=TensorDRAMLayout.HWC,
    )
    tessellate_parameters = {
        input_name: dataclasses.replace(input_parameters)
        for input_name in mla_node.input_names
    }

    output_parameters = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0),
        enable_mla=True,
        dram_layout=TensorDRAMLayout.HWC16,
    )
    output_node = mla_node.ir.nodes[mla_node.ir.output_node_name]
    output_names = (
        output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    )
    for output_name in output_names:
        tessellate_parameters[f"{output_name}_output"] = dataclasses.replace(
            output_parameters
        )

    print(f"segments={segments}", flush=True)
    print(f"batch_size={args.batch_size}", flush=True)
    print(f"tessellated_inputs={list(mla_node.input_names)}", flush=True)
    print(f"tessellated_outputs={output_names}", flush=True)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.compiler_debug_directory.parent.mkdir(parents=True, exist_ok=True)
    model.compile(
        output_path=str(args.output_directory),
        batch_size=args.batch_size,
        compress=True,
        preserve=True,
        deployable=False,
        tessellate_parameters=tessellate_parameters,
        retained_temporary_directory_name=str(args.compiler_debug_directory),
        log_level=logging.INFO,
    )
    print(f"output_directory={args.output_directory.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
