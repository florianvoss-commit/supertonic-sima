# Supertonic 3 on SiMa Modalix

This repository makes the Supertonic 3 vector estimator suitable for a fixed
SiMa Modalix profile, validates it against the released ONNX model, and
documents the BF16-activation/INT8-weight compilation flow.

The current compiled model is one `MLA_0` compute partition. It is not split
into the 58 partitions produced by the original graph. FP32/BF16 casts and the
I/O packing transform remain on EV74 as expected.

## Deployment split

| Component | Calls per utterance | Initial target |
|---|---:|---|
| Text preprocessing | 1 | A65 |
| Duration predictor | 1 | A65 / ONNX Runtime |
| Text encoder | 1 | A65 / ONNX Runtime |
| Vector field | 2 branches × 8 steps | Modalix MLA |
| CFG and Euler update | 8 | A65 |
| Vocoder | 1 | A65 / ONNX Runtime |

The vector estimator represented about 90% of measured CPU inference time for
a 12.66-second test utterance. The vocoder is the next candidate for MLA only
if measurement on the target A65 shows that it cannot meet the latency goal.

## Fixed compiled contract

All boundary tensors are FP32. Computation uses BF16 activations and symmetric
per-channel INT8 weights.

| Index | Input | Shape |
|---:|---|---|
| 0 | `noisy_latent` | `[1, 144, 1, 192]` |
| 1 | `text_emb` | `[1, 256, 1, 192]` |
| 2 | `style_ttl` | `[1, 256, 1, 50]` |
| 3 | `style_key` | `[1, 256, 1, 50]` |
| 4 | `latent_mask` | `[1, 1, 1, 192]` |
| 5 | `text_mask` | `[1, 192, 1, 1]` |
| 6 | `time_sinusoidal` | `[1, 64, 1, 1]` |
| 7 | `rope_tables` | `[1, 128, 1, 192]` |

Output: `velocity`, shape `[1, 144, 1, 192]`, FP32.

The host selects `time_sinusoidal` by denoising step and selects the two RoPE
sin/cos pairs by the effective latent and text-mask lengths. The host invokes
the same MLA graph for the conditional and unconditional branches, then applies
classifier-free guidance and the Euler update.

## Pinned upstream

```text
repository: Supertone/supertonic-3
revision:   724fb5abbf5502583fb520898d45929e62f02c0b
package:    supertonic==1.3.1
vector estimator SHA-256:
883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c
```

Generated weights, ONNX files, NPZ files, WAV files, compiler output, and model
archives are intentionally ignored by Git. The compiled release belongs in
[`florianvoss/supertonic-3-sima`](https://huggingface.co/florianvoss/supertonic-3-sima).
Recorded hashes and validation metrics are in
[`artifacts/manifest.json`](artifacts/manifest.json).

## 1. Prepare the graph-surgery environment

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

Download the pinned upstream model:

```bash
mkdir -p models
.venv/bin/hf download Supertone/supertonic-3 \
  --revision 724fb5abbf5502583fb520898d45929e62f02c0b \
  --local-dir models/supertonic-3

echo '883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c  models/supertonic-3/onnx/vector_estimator.onnx' \
  | sha256sum --check
```

## 2. Generate deterministic references

Reference generation is isolated from the graph-surgery environment because
it uses the released Supertonic package.

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

## 3. Run graph surgery

`graph_surgery.py` is the single public graph-preparation command. It
staticizes the wrapper, extracts the batch-one vector field, lifts data tensors
to rank four, applies MLA-oriented attention/padding rewrites, externalizes the
timestep and RoPE calculations, and validates all five reference cases.

```bash
.venv/bin/python tools/graph_surgery.py \
  --source models/supertonic-3/onnx/vector_estimator.onnx \
  --reference-dir testdata/reference_cases \
  --output-dir build/surgery \
  --steps 8
```

Final outputs:

| Artifact | SHA-256 |
|---|---|
| `supertonic_vector_field_sima.onnx` | `79f6c51e73f4c8c20450fcede73fccbc183f4b22e5e20b78d64bc3aa585def5a` |
| `supertonic_runtime_data.npz` | `81fc7a131dd0eafe6fa7062b23f49e0dc9e3fb178f5473bf8d404ab555bcf5aa` |

The final graph contains 1,080 nodes and no live `Sin`, `Cos`, `Gather`,
`Squeeze`, `Unsqueeze`, `Equal`, `Where`, `Reshape`, or `Identity` nodes. The
five-case, eight-step, two-branch surgery gate performs 80 comparisons and is
bit-exact against the pre-externalized graph.

## 4. Audit, quantize, and compile

Use the existing `afe_env` without installing or changing packages. The SiMa
model-compiler skills must already be installed.

```bash
source ../afe_env/bin/activate

python /home/florian.voss/.codex_sw-u24-llm/skills/model_surgery/scripts/model_surgery_guard.py \
  audit-model --model build/surgery/supertonic_vector_field_sima.onnx \
  --dtype int8 --json
```

Quantize with the standard skill driver and random calibration. `--no-compile`
keeps packaging separate from quantization verification and avoids the known
release-2.1 `mpk2conf` post-processing issue.

```bash
python /home/florian.voss/.codex_sw-u24-llm/skills/quantize_compile/scripts/quantize_compile.py \
  --model_path build/surgery/supertonic_vector_field_sima.onnx \
  --model_format onnx \
  --model_layout NCHW \
  --input_names noisy_latent text_emb style_ttl style_key latent_mask text_mask time_sinusoidal rope_tables \
  --input_shapes 1,144,1,192 1,256,1,192 1,256,1,50 1,256,1,50 1,1,1,192 1,192,1,1 1,64,1,1 1,128,1,192 \
  --output_names velocity \
  --device modalix \
  --build_dir build/modalix-bf16-int8w \
  --no-simplify \
  --bf16-activations \
  --any_shape_on_mla \
  --verify \
  --executor jax \
  --no-compile
```

Compile the saved quantized network with MLA I/O tessellation and retained
compiler output:

```bash
python tools/compile_saved_tessellated.py \
  --network-directory build/modalix-bf16-int8w/supertonic_vector_field_sima \
  --model-name supertonic_vector_field_sima \
  --output-directory build/modalix-bf16-int8w/compiled \
  --compiler-debug-directory build/modalix-bf16-int8w/compiler-debug
```

The recorded MPK contains one MLA compute plugin, eight FP32-to-BF16 input
casts, one input pack transform, and one output cast. It contains no A65 compute
partition.

## 5. Run without the Supertonic package

`run_sima_onnx.py` needs only NumPy, ONNX Runtime, the four upstream CPU model
files/voice styles, the surgically rewritten vector field, and its runtime NPZ.
It does not import the `supertonic` package.

```bash
.venv/bin/python tools/run_sima_onnx.py \
  --model-dir models/supertonic-3 \
  --vector-field build/surgery/supertonic_vector_field_sima.onnx \
  --runtime-data build/surgery/supertonic_runtime_data.npz \
  --text 'Hello from Modalix.' \
  --voice M1 \
  --lang en \
  --steps 8 \
  --speed 1.0 \
  --seed 1101 \
  --output build/hello.wav \
  --report build/hello.json
```

An utterance must fit both static limits: processed text length at most 192 and
predicted latent length at most 192 (about 13.37 seconds). A live assistant
should split at sentence boundaries, fall back to clause boundaries for long
sentences, and join chunks with silence or a short crossfade.

## 6. Validate the quantized production loop

This optional AFE validation replaces only the vector-estimator session inside
the released production loop. It generates ONNX and quantized WAVs and records
every denoising step.

```bash
source ../afe_env/bin/activate
python tools/verify_production_quantized.py \
  --supertonic-site-packages build/production-loop-venv/lib/python3.12/site-packages \
  --model-dir models/supertonic-3 \
  --source-vector-estimator models/supertonic-3/onnx/vector_estimator.onnx \
  --quantized-dir build/modalix-bf16-int8w/supertonic_vector_field_sima \
  --quantized-name supertonic_vector_field_sima \
  --runtime-data build/surgery/supertonic_runtime_data.npz \
  --reference-case testdata/reference_cases/en_m1_short.npz \
  --text 'Hello from Neat GenAI Studio on the Modalix DevKit.' \
  --voice M1 --lang en --steps 8 --speed 1.0 --seed 1101 \
  --output-dir build/validation/en_m1_short
```

For perceptual diagnostics, compare generated WAVs with
`tools/analyze_mel_distance.py`; direct waveform cosine is very sensitive to
small phase and timing shifts.

## Repository layout

- `tools/graph_surgery.py` — main graph-surgery entry point
- `tools/run_sima_onnx.py` — package-free ONNX production runtime
- `tools/compile_saved_tessellated.py` — retained one-MLA packaging step
- `tools/verify_production_quantized.py` — real eight-step AFE validation
- `tools/analyze_mel_distance.py` — multi-resolution log-mel comparison
- `tools/model_contract.py` — source, pre-external, and final tensor contracts
- `artifacts/manifest.json` — recorded compilation and validation provenance
- `IMPLEMENTATION_PLAN.md` — current status and remaining deployment work
