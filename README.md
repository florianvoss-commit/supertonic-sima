# Supertonic 3 on SiMa Modalix

This repository makes the Supertonic 3 vector estimator and vocoder suitable
for a fixed SiMa Modalix profile, validates them against the released ONNX
models, and documents both compiled precision profiles.

The vector field uses BF16 activations with INT8 weights; the vocoder uses BF16
activations and weights to preserve its quantization-sensitive output head.
Each compiled model has one `MLA_0` compute partition instead of the 58
partitions produced by the original graph. FP32/BF16 boundary casts and vector
field I/O packing remain on EV74 as expected.

## Deployment split

| Component | Calls per utterance | Initial target |
|---|---:|---|
| Text preprocessing | 1 | A65 |
| Duration predictor | 1 | A65 / ONNX Runtime |
| Text encoder | 1 | A65 / ONNX Runtime |
| Vector field | 2 branches × 8 steps | Modalix MLA |
| CFG and Euler update | 8 | A65 |
| Vocoder | 1 | Modalix MLA; A65 / ONNX Runtime fallback |

The vector estimator and full-BF16 vocoder are compiled for MLA. The CPU models
and host-side CFG/Euler update remain unchanged.

## Fixed compiled contract

All boundary tensors are FP32. The vector field uses BF16 activations and
symmetric per-channel INT8 weights. The vocoder uses full BF16 internally.

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

The fixed vocoder contract accepts the vector field's 4D latent directly:

| Index | Input | Shape |
|---:|---|---|
| 0 | `latent` | `[1, 144, 1, 192]` |

Vocoder output: `wav_frames`, logical NCHW shape `[1, 512, 1, 1152]`, FP32.
With HWC/HWC16 output tessellation, the raw bytes are time-major 512-sample
frames and can be read as one 589,824-sample waveform without a host transpose.

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
vocoder SHA-256:
085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba
```

Generated weights, ONNX files, NPZ files, WAV files, compiler output, and model
archives are intentionally ignored by Git. The compiled release belongs in
[`florianvoss/supertonic-3-sima`](https://huggingface.co/florianvoss/supertonic-3-sima).
Recorded hashes and validation metrics are in
[`artifacts/manifest.json`](artifacts/manifest.json) and
[`artifacts/vocoder_bf16_manifest.json`](artifacts/vocoder_bf16_manifest.json).

## 1. Prepare the graph-surgery environment

Python 3.12 was used for the recorded artifacts.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  numpy==2.5.2 \
  onnx==1.17.0 \
  onnxruntime==1.22.1 \
  'onnxsim>=0.4.36,<0.5' \
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
echo '085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba  models/supertonic-3/onnx/vocoder.onnx' \
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
timestep and RoPE calculations, and validates all five reference cases. It also
staticizes the vocoder, converts Conv1D to Conv2D, replaces its input pixel
shuffle with `DepthToSpace`, and exposes native HWC-readable waveform frames.

```bash
.venv/bin/python tools/graph_surgery.py \
  --source models/supertonic-3/onnx/vector_estimator.onnx \
  --vocoder-source models/supertonic-3/onnx/vocoder.onnx \
  --reference-dir testdata/reference_cases \
  --output-dir build/surgery \
  --steps 8
```

Final outputs:

| Artifact | SHA-256 |
|---|---|
| `supertonic_vector_field_sima.onnx` | `79f6c51e73f4c8c20450fcede73fccbc183f4b22e5e20b78d64bc3aa585def5a` |
| `supertonic_runtime_data.npz` | `81fc7a131dd0eafe6fa7062b23f49e0dc9e3fb178f5473bf8d404ab555bcf5aa` |
| `supertonic_vocoder_sima.onnx` | `b6d233bb3cd6062f613fcc2e8b9192b1447408db06995b215f1da0d66f1a9684` |

The final graph contains 1,080 nodes and no live `Sin`, `Cos`, `Gather`,
`Squeeze`, `Unsqueeze`, `Equal`, `Where`, `Reshape`, or `Identity` nodes. The
five-case, eight-step, two-branch surgery gate performs 80 comparisons and is
bit-exact against the pre-externalized graph.

The vocoder graph contains no `Reshape` nodes. Its two non-LayerNorm
transposes are removed, leaving only the 20 semantic transpose nodes surrounding
the ten channel-wise LayerNorm operations. One `DepthToSpace(CRD, blocksize=6)`
replaces the input channel-to-time shuffle, and all 12 edge pads are expressed
as exact static `Slice`/`Concat` replication. Every data activation is rank
four. Two waveform comparisons reached cosine `0.9999999999678247` or better,
with maximum relative L2 error `8.03e-6`.

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

The recorded vector-field MPK contains one MLA compute plugin, eight
FP32-to-BF16 input casts, one input pack transform, and one output cast. It
contains no A65 compute partition.

### Full-BF16 vocoder

The vocoder's INT8 weights cause a sparse positive activation mask to collapse
before its near-zero-slope PReLU. Full BF16 avoids that catastrophic failure.
Quantize and compile the prepared vocoder with:

```bash
python /home/florian.voss/.codex_sw-u24-llm/skills/quantize_compile/scripts/quantize_compile.py \
  --model_path build/surgery/supertonic_vocoder_sima.onnx \
  --model_format onnx --model_layout NCHW \
  --input_names latent --input_shapes 1,144,1,192 \
  --output_names wav_frames --device modalix \
  --build_dir build/vocoder-modalix-bf16-bf16w \
  --no-simplify --bf16-activations --bf16-weights \
  --any_shape_on_mla --mla-tesselation \
  --verify --executor jax --no-compile

python tools/compile_saved_tessellated.py \
  --network-directory build/vocoder-modalix-bf16-bf16w/supertonic_vocoder_sima \
  --model-name supertonic_vocoder_sima \
  --output-directory build/vocoder-modalix-bf16-bf16w/compiled \
  --compiler-debug-directory build/vocoder-modalix-bf16-bf16w/compiler-debug
```

The resulting MPK has one MLA compute plugin, two expected EV74 boundary casts,
and no A65 compute partition. Its compiler cycle estimate is `11,625,345`.
It is published as `supertonic_vocoder_sima_bf16_mpk.tar.gz` in the Hugging
Face repository linked above.
See [`artifacts/vocoder_bf16_manifest.json`](artifacts/vocoder_bf16_manifest.json)
for hashes and validation details.

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

For the vocoder, full BF16 improves cropped waveform cosine from
`0.420`–`0.444` with INT8 weights to `0.992`–`0.996`. Relative L2 improves from
`0.897`–`0.909` to `0.093`–`0.125`. The saved FP32 and quantized AFE graphs were
also compared across all 178 analyzable layers, independently of ONNX/AFE
layout conversion.

## Repository layout

- `tools/graph_surgery.py` — main graph-surgery entry point
- `tools/optimize_vocoder_4d.py` — static all-4D vocoder transformation
- `tools/run_sima_onnx.py` — package-free ONNX production runtime
- `tools/compile_saved_tessellated.py` — retained one-MLA packaging step
- `tools/verify_production_quantized.py` — real eight-step AFE validation
- `tools/verify_vocoder_quantized.py` — exact saved-vocoder AFE/ONNX comparison
- `tools/analyze_vocoder_quantization.py` — cumulative and local-feed layer-error tracing
- `tools/analyze_mel_distance.py` — multi-resolution log-mel comparison
- `tools/model_contract.py` — source, pre-external, and final tensor contracts
- `artifacts/manifest.json` — recorded compilation and validation provenance
- `artifacts/vocoder_bf16_manifest.json` — full-BF16 vocoder provenance and validation
- `IMPLEMENTATION_PLAN.md` — current status and remaining deployment work
