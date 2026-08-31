# Supertonic 3 on SiMa Modalix

This repository contains the reproducible graph-surgery and verification tools
for compiling the Supertonic 3 vector estimator for SiMa Modalix. It targets one
fixed profile:

- batch size: `1`
- text length: `192`
- latent length: `192`
- public and internal floating-point tensors: rank 4 with leading batch `1`
- target precision: BF16 activations and BF16 weights

The duration predictor, text encoder, text processing, voice-style loading, and
vocoder remain CPU ONNX Runtime components. Only the denoising/vector-estimator
core is intended for Modalix MLA.

The detailed design, acceptance gates, and current status are in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). Phases 1–3—graph surgery,
reference generation, and ONNX numerical verification—are complete. Phase 4,
BF16 import/quantization/compilation, is the next step.

## Repository policy

No model weights or generated model artifacts are committed. In particular,
Git ignores ONNX, PTH/PT, safetensors, NPZ, WAV, MPK, compiler archives, virtual
environments, and the complete `build/` directory.

The released model is downloaded from its pinned Hugging Face revision, and all
compiler-facing ONNX files are recreated locally by the checked-in tools.

## Pinned upstream model

```text
repository: Supertone/supertonic-3
revision:   724fb5abbf5502583fb520898d45929e62f02c0b
package:    supertonic==1.3.1
```

Expected source vector-estimator SHA-256:

```text
883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c
```

## 1. Create the graph-surgery environment

Python 3.12 was used for the recorded artifacts.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  numpy==2.5.2 \
  onnx==1.17.0 \
  onnxruntime==1.22.1 \
  huggingface_hub
```

## 2. Download the pinned upstream files

Authenticate first if the local Hugging Face configuration is not already
authenticated:

```bash
hf auth login
```

Download the complete pinned model. The CPU components are needed to regenerate
reference cases, while graph surgery uses `onnx/vector_estimator.onnx`.

```bash
mkdir -p models
.venv/bin/hf download Supertone/supertonic-3 \
  --revision 724fb5abbf5502583fb520898d45929e62f02c0b \
  --local-dir models/supertonic-3
```

Verify the source model before running surgery:

```bash
echo '883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c  models/supertonic-3/onnx/vector_estimator.onnx' \
  | sha256sum --check
```

## 3. Recreate the compiler-facing ONNX

```bash
mkdir -p build/onnx build/reports

.venv/bin/python tools/staticize_vector_estimator.py \
  --input models/supertonic-3/onnx/vector_estimator.onnx \
  --output build/onnx/vector_estimator_b1_t192_l192.static.onnx \
  --report build/reports/phase1_staticize.json

.venv/bin/python tools/extract_vector_field.py \
  --source models/supertonic-3/onnx/vector_estimator.onnx \
  --output build/onnx/vector_field_b1_t192_l192.rank3.onnx \
  --report build/reports/phase1_extract_vector_field.json

.venv/bin/python tools/lift_vector_field_to_4d.py \
  --source build/onnx/vector_field_b1_t192_l192.rank3.onnx \
  --output build/onnx/vector_field_b1_t192_l192.all4d.onnx \
  --report build/reports/phase1_lift_all4d.json

.venv/bin/python tools/optimize_vector_field_4d.py \
  --source build/onnx/vector_field_b1_t192_l192.all4d.onnx \
  --output build/onnx/vector_field_b1_t192_l192.all4d.opt.onnx \
  --report build/reports/phase1_optimize_all4d.json
```

Expected hashes from the recorded environment:

| Artifact | SHA-256 |
|---|---|
| Static wrapper | `062c0a3680bd46e640b80dbfe0a67dfaa8d6d4d41e032855961199e45e44d917` |
| Extracted rank-3 core | `b0ad3521b2190a81f039dd3bf25bfc38704218b856f66427f3ceee6665220e62` |
| Initial all-4D core | `a83801537fa78ffd277b60e38a6f918405add83bfbcd3344dafe690a79413d1e` |
| Final optimized all-4D core | `00d9562f94450822cc862ed358b179531622e433fbdd072dc6f9ef4a8ee41a51` |

The final graph has no `Identity`, `Reshape`, `Gather`, `Squeeze`, `Unsqueeze`,
`Equal`, or `Where` nodes. Its 72 remaining `Transpose` nodes are exclusively
the two layout adapters around each of 36 `LayerNormalization` nodes. Attention
uses the same channel-first `[B,D,H,L]` and two-Einsum structure as SiMa LLiMa's
optimized cache attention.

## 4. Regenerate deterministic reference cases

Use a separate environment so the released Supertonic package cannot disturb
the compiler or graph-surgery dependency set.

```bash
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install --upgrade pip
.venv-reference/bin/python -m pip install supertonic==1.3.1

mkdir -p testdata/reference_cases

.venv-reference/bin/python tools/generate_reference_cases.py \
  --model-dir models/supertonic-3 \
  --output-dir testdata/reference_cases \
  --summary testdata/manifest.json
```

Generated NPZ and WAV files remain local. `testdata/manifest.json` records their
seeds, tensor shapes, component hashes, and expected artifact hashes.

## 5. Re-run numerical verification

```bash
.venv/bin/python tools/verify_onnx_equivalence.py \
  --source models/supertonic-3/onnx/vector_estimator.onnx \
  --static build/onnx/vector_estimator_b1_t192_l192.static.onnx \
  --candidate build/onnx/vector_field_b1_t192_l192.all4d.opt.onnx \
  --reference-dir testdata/reference_cases \
  --report build/reports/phase3_onnx_equivalence.json
```

This checks source-versus-static execution, fixed-width padding behavior, and
teacher-forced/free-running classifier-free-guidance denoising for 1, 2, and 8
steps. The graph includes mask-aware edge filling before all 28 ConvNeXt edge
pads; naive zero-padding of the released dynamic model is not numerically
equivalent at the natural latent boundary.

## 6. Continue with the installed Model Compiler

On a machine where the SiMa Model Compiler is already installed:

```bash
activate-model-compiler
python -c 'import afe; print(afe.__file__)'
```

Continue at Phase 4 in `IMPLEMENTATION_PLAN.md`, using:

```text
build/onnx/vector_field_b1_t192_l192.all4d.opt.onnx
testdata/reference_cases/*.npz
```

The compiler driver must preserve the ordered eight-input FP32 boundary from
`tools/model_contract.py`, select `gen2_target`, use `bfloat16_scheme()` for
both activations and weights, execute FP and BF16 models on representative
Supertonic tensors before compilation, and record compiler/package versions and
all hashes. Do not use an image-oriented helper that converts every input from
NCHW to NHWC.

## Key files

- `IMPLEMENTATION_PLAN.md` — implementation phases, contracts, findings, and gates
- `tools/model_contract.py` — pinned revisions, hashes, and ordered tensor contract
- `tools/extract_vector_field.py` — extracts the true batch-1 vector field
- `tools/lift_vector_field_to_4d.py` — converts every floating activation to rank 4
- `tools/optimize_vector_field_4d.py` — LLiMa-style attention and boundary surgery
- `tools/generate_reference_cases.py` — deterministic CPU reference generation
- `tools/verify_onnx_equivalence.py` — complete ONNX numerical verification
