#!/usr/bin/env python3
"""Locate cumulative and per-operator quantization error in the saved vocoder."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


LATENT_CHANNELS = 144
LATENT_LENGTH = 192


def comparison(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        return {
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "shape_mismatch": True,
        }
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    ref64 = reference.astype(np.float64)
    actual64 = actual.astype(np.float64)
    ref_l2 = float(np.linalg.norm(ref64.ravel()))
    actual_l2 = float(np.linalg.norm(actual64.ravel()))
    denom = ref_l2 * actual_l2
    reference_positive = ref64 > 0
    actual_positive = actual64 > 0
    positive_union = int(np.count_nonzero(reference_positive | actual_positive))
    return {
        "shape": list(reference.shape),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / max(ref_l2, 1e-30)),
        "cosine": float(np.dot(ref64.ravel(), actual64.ravel()) / denom)
        if denom
        else 1.0,
        "reference_rms": float(np.sqrt(np.mean(np.square(ref64)))),
        "actual_rms": float(np.sqrt(np.mean(np.square(actual64)))),
        "reference_mean": float(np.mean(ref64)),
        "actual_mean": float(np.mean(actual64)),
        "mean_shift": float(np.mean(actual64) - np.mean(ref64)),
        "reference_min": float(np.min(ref64)),
        "reference_max": float(np.max(ref64)),
        "actual_min": float(np.min(actual64)),
        "actual_max": float(np.max(actual64)),
        "reference_positive_fraction": float(np.mean(reference_positive)),
        "actual_positive_fraction": float(np.mean(actual_positive)),
        "positive_mask_disagreement": float(
            np.mean(reference_positive != actual_positive)
        ),
        "positive_mask_iou": float(
            np.count_nonzero(reference_positive & actual_positive) / positive_union
        )
        if positive_union
        else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized-dir", type=Path, required=True)
    parser.add_argument("--quantized-name", required=True)
    parser.add_argument("--reference-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("global", "local", "both"), default="both"
    )
    args = parser.parse_args()

    from afe.apis.model import Model
    from afe.core.graph_analyzer.graph_analyzer import QuantizedGraphAnalyzer

    with np.load(args.reference_case) as archive:
        latent = np.asarray(archive["final_latent_natural"], dtype=np.float32)
    if latent.shape[0:2] != (1, LATENT_CHANNELS):
        raise ValueError(f"unexpected latent shape: {latent.shape}")
    if latent.shape[-1] > LATENT_LENGTH:
        raise ValueError(f"latent is too long: {latent.shape}")
    latent = np.pad(latent, ((0, 0), (0, 0), (0, LATENT_LENGTH - latent.shape[-1])))
    inputs = {"latent": np.ascontiguousarray(latent[:, :, None, :].transpose(0, 2, 3, 1))}

    model = Model.load(
        model_name=args.quantized_name,
        network_directory=str(args.quantized_dir.resolve()),
        include_unquantized_net=True,
        log_level=logging.WARNING,
    )
    if model._fp32_net is None:
        raise RuntimeError("saved model has no paired floating-point graph")

    nodes = {node.name: node for node in model._net.iter_nodes_recursive()}
    order = {node.name: index for index, node in enumerate(model._net.iter_nodes_recursive())}
    analyzer = QuantizedGraphAnalyzer()
    print("Executing paired AFE floating-point graph...", flush=True)
    reference = analyzer._execute_net(model._fp32_net, inputs)

    result: dict[str, Any] = {
        "reference_case": str(args.reference_case.resolve()),
        "input_shape": list(inputs["latent"].shape),
        "mode": args.mode,
    }

    def build_rows(values: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, value in values.items():
            ref_value = reference.get(name)
            if not isinstance(value, np.ndarray) or not isinstance(ref_value, np.ndarray):
                continue
            node = nodes.get(name)
            operation = None
            if node is not None and hasattr(node.ir, "operation"):
                operation = type(node.ir.operation).__name__
            rows.append(
                {
                    "index": order.get(name),
                    "node": name,
                    "operation": operation,
                    **comparison(ref_value, value),
                }
            )
        return sorted(rows, key=lambda row: row["index"] if row["index"] is not None else -1)

    if args.mode in ("global", "both"):
        print("Executing cumulative/global quantized graph...", flush=True)
        global_values = analyzer._execute_net(
            model._net, inputs, dequantize_intermediate=True
        )
        global_rows = build_rows(global_values)
        result["global"] = global_rows
        print(f"Collected {len(global_rows)} global layer comparisons.", flush=True)

    if args.mode in ("local", "both"):
        print("Executing isolated/local-feed quantized operators...", flush=True)
        local_values = analyzer._execute_quantized_net_with_local_inputs(
            model._net,
            model._fp32_net,
            inputs,
            reference,
            dequantize_intermediate=True,
        )
        local_rows = build_rows(local_values)
        result["local"] = local_rows
        print(f"Collected {len(local_rows)} local layer comparisons.", flush=True)

    for mode in ("global", "local"):
        rows = result.get(mode)
        if not rows:
            continue
        result[f"{mode}_summary"] = {
            "last_layer": rows[-1],
            "largest_relative_l2": sorted(
                rows, key=lambda row: row.get("relative_l2", -1), reverse=True
            )[:15],
            "largest_mae": sorted(
                rows, key=lambda row: row.get("mae", -1), reverse=True
            )[:15],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
