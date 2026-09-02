#!/usr/bin/env python3
"""Build the final static SiMa vector-field ONNX with one command.

The specialized transformation modules remain independently testable, but this
is the public graph-surgery entry point.  It extracts the batch-one vector
field, lifts all data activations to rank four, applies the MLA-oriented graph
rewrites, and externalizes timestep/RoPE values and CFG constants.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from extract_vector_field import extract
from lift_vector_field_to_4d import lift
from model_contract import SOURCE_SHA256
from optimize_vector_field_4d import optimize


DEFAULT_CASES = (
    "en_m1_short",
    "en_m1_medium",
    "en_f1_short",
    "es_m1_short",
    "en_m1_near_capacity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Pinned upstream onnx/vector_estimator.onnx.",
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--validation-cases",
        nargs="+",
        choices=DEFAULT_CASES,
        default=list(DEFAULT_CASES),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    intermediate_dir = output_dir / "intermediate"
    reports_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {source_hash}"
        )

    references = [reference_dir / f"{name}.npz" for name in args.validation_cases]
    missing = [str(path) for path in references if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation cases: {missing}")

    static_wrapper = intermediate_dir / "vector_estimator.static.onnx"
    rank3 = intermediate_dir / "vector_field.rank3.onnx"
    all4d = intermediate_dir / "vector_field.all4d.onnx"
    optimized = intermediate_dir / "vector_field.all4d.opt.onnx"
    final_model = output_dir / "supertonic_vector_field_sima.onnx"
    runtime_data = output_dir / "supertonic_runtime_data.npz"
    externalize_report = reports_dir / "externalize_sinusoidal_inputs.json"

    report: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": source_hash},
        "steps": args.steps,
        "validation_cases": list(args.validation_cases),
        "stages": {},
    }
    report_path = output_dir / "graph_surgery.json"
    write_json(report_path, report)

    staticizer = Path(__file__).with_name("staticize_vector_estimator.py")
    staticize_report = reports_dir / "staticize.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(staticizer),
            "--input",
            str(source),
            "--output",
            str(static_wrapper),
            "--report",
            str(staticize_report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "staticization failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    report["stages"]["staticize"] = json.loads(staticize_report.read_text())

    # Extract from the static wrapper. Extracting directly from the symbolic
    # source would make the rank lifter conservatively map unknown lengths to
    # one, which is not the fixed 192-position deployment contract.
    report["stages"]["extract"] = extract(static_wrapper, rank3)
    write_json(reports_dir / "extract.json", report["stages"]["extract"])
    report["stages"]["lift_all4d"] = lift(rank3, all4d)
    write_json(reports_dir / "lift_all4d.json", report["stages"]["lift_all4d"])
    report["stages"]["optimize"] = optimize(all4d, optimized)
    write_json(reports_dir / "optimize.json", report["stages"]["optimize"])

    externalizer = Path(__file__).with_name("externalize_sinusoidal_tables.py")
    command = [
        sys.executable,
        str(externalizer),
        "--input",
        str(optimized),
        "--output",
        str(final_model),
        "--table",
        str(runtime_data),
        "--report",
        str(externalize_report),
        "--source-wrapper",
        str(source),
        "--reference-case",
        *(str(path) for path in references),
        "--steps",
        str(args.steps),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "sinusoidal-input surgery failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    report["stages"]["externalize"] = json.loads(
        externalize_report.read_text()
    )
    report["status"] = "passed"
    report["artifacts"] = {
        "model": {
            "path": str(final_model),
            "sha256": sha256(final_model),
        },
        "runtime_data": {
            "path": str(runtime_data),
            "sha256": sha256(runtime_data),
        },
        "report": str(report_path),
    }
    write_json(report_path, report)
    validation = report["stages"]["externalize"]["validation"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": report["artifacts"]["model"],
                "runtime_data": report["artifacts"]["runtime_data"],
                "validation": {
                    "comparisons": len(validation["comparisons"]),
                    "max_abs": validation["max_abs"],
                    "max_relative_l2": validation["max_relative_l2"],
                    "min_cosine": validation["min_cosine"],
                },
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
