# Supertonic 3 Modalix deployment plan

## Goal

Run Supertonic 3 speech synthesis with its iterative vector field and vocoder
on Modalix MLA, with preprocessing and the smaller models on the embedded A65.
Use sentence-sized chunks and a fixed 192-position MLA profile.

## Current status

| Work item | Status |
|---|---|
| Pin upstream model and deterministic references | Complete |
| Staticize and extract the vector field | Complete |
| Convert live activations to rank four | Complete |
| Rewrite attention, masks, and edge padding for MLA | Complete |
| Externalize timestep and length-dependent RoPE values | Complete |
| Validate graph surgery across five cases and eight steps | Complete |
| Quantize BF16 activations / INT8 weights | Complete |
| Compile as one tessellated MLA segment | Complete |
| Validate real eight-step quantized production loop | Complete |
| Staticize and convert vocoder to all-4D MLA graph | Complete |
| Validate vocoder surgery against released ONNX | Complete |
| Quantize and compile vocoder on MLA | Complete; full BF16 selected |
| Run compiled MPK on Modalix hardware | Pending |
| Integrate a persistent `pyneat` worker | Pending |
| Publish refreshed Hugging Face artifacts | Complete for vector field and BF16 vocoder |

## Fixed profile and interface

- Batch: `1`
- Maximum processed text length: `192`
- Maximum latent length: `192`
- Maximum audio duration: approximately `13.37 s`
- Boundary dtype: FP32
- Vector-field precision: BF16 activations, symmetric per-channel INT8 weights
- Vocoder precision: BF16 activations and weights
- Denoising steps: `8`
- Branches per step: conditional and unconditional

Final ordered input contract:

| Index | Name | Shape |
|---:|---|---|
| 0 | `noisy_latent` | `[1, 144, 1, 192]` |
| 1 | `text_emb` | `[1, 256, 1, 192]` |
| 2 | `style_ttl` | `[1, 256, 1, 50]` |
| 3 | `style_key` | `[1, 256, 1, 50]` |
| 4 | `latent_mask` | `[1, 1, 1, 192]` |
| 5 | `text_mask` | `[1, 192, 1, 1]` |
| 6 | `time_sinusoidal` | `[1, 64, 1, 1]` |
| 7 | `rope_tables` | `[1, 128, 1, 192]` |

Output: `velocity`, `[1, 144, 1, 192]`, FP32.

Vocoder input: `latent`, `[1, 144, 1, 192]`, FP32.

Vocoder output: `wav_frames`, `[1, 512, 1, 1152]`, FP32. With HWC/HWC16
tessellation its bytes are already in time-major waveform order.

The previous internal `current_step` and `total_step` inputs are not part of
the final contract. The host selects one precomputed timestep row and packs
latent/text RoPE pairs selected by effective mask length.

## Completed graph work

`tools/graph_surgery.py` is the supported public entry point. Its internal
stages:

1. Check the pinned source SHA-256 and freeze batch/text/latent dimensions.
2. Extract the batch-one vector field from the released CFG/Euler wrapper.
3. Lift floating data activations to the all-rank-four compiler contract.
4. Rewrite attention to channel-first MLA-friendly operations.
5. Replace `Where(-inf)` masks with finite FP32-min arithmetic masks.
6. Make ConvNeXt padding mask-aware, then decompose edge padding into supported
   zero padding plus exact boundary corrections.
7. Externalize timestep sinusoidal values and length-dependent RoPE pairs.
8. Run ONNX checking, strict shape inference, and numerical comparison.

The final ONNX has 1,080 nodes. The externalization gate covers five reference
cases, eight steps, and both branches: 80 comparisons with zero observed
difference from the pre-externalized graph.

The same entry point optionally processes the vocoder. It replaces the input
Reshape/Transpose/Reshape pixel shuffle with one supported
`DepthToSpace(CRD, blocksize=6)`, lifts all Conv1D operations to Conv2D, removes
the final waveform transpose/flatten, and converts all 12 edge pads to exact
static Slice/Concat replication. The result has zero Reshape nodes and all data
activations are rank four. The 20 remaining Transposes are the semantic pairs
around the ten channel-wise LayerNorm operations. Two real-latent waveform
comparisons pass with maximum relative L2 error `8.03e-6` and minimum cosine
`0.9999999999678247`.

## Completed compilation

Recorded environment and configuration:

- Existing `afe_env`, unchanged
- Model Compiler `2.1.3`
- Modalix `gen2_target`
- Automatic random calibration
- BF16 activations
- Symmetric per-channel INT8 weights
- `any_shape_on_mla=True`
- MLA input/output tessellation enabled
- Compression enabled

The quantized network contains exactly `MLA_0`. The final MPK plugin sequence is
eight input casts, one pack transform, `MLA_0`, and one output cast. EV74 owns
only those boundary helpers; no model computation is partitioned to A65/APU.

Artifact hashes, sizes, and validation metrics are recorded in
`artifacts/manifest.json`.

The full-BF16 vocoder also compiles into exactly one `MLA_0` partition with one
input cast, one output cast, and no A65/APU compute partition. Its cycle
estimate is `11,625,345`. Separate provenance is recorded in
`artifacts/vocoder_bf16_manifest.json`.

## Accuracy findings

Graph surgery itself is exact. The remaining error is mixed-precision
quantization error accumulated by the free-running denoising loop.

Short English case (`4.14 s`):

- Step-1 cosine: `0.999926`
- Step-8/final-latent cosine: `0.979563`
- Waveform cosine: `0.835558`
- SI-SDR: `3.64 dB`

Near-capacity English case (`12.66 s`, latent length 182):

- Step-1 cosine: `0.999931`
- Step-8/final-latent cosine: `0.968573`
- Waveform cosine: `0.479883`
- SI-SDR: `-5.24 dB`

Listening indicates substantially smaller perceptual degradation than the
sample-aligned waveform metrics suggest. Small timing/phase deviations are
strongly penalized by waveform cosine and SI-SDR. Log-mel distance and listening
tests should remain the primary audio acceptance evidence.

## Runtime architecture

The first application version should use:

1. A65 text normalization and Unicode indexing.
2. A65 ONNX Runtime duration prediction and text encoding.
3. A65 allocation/padding of the fixed input buffers.
4. Persistent `pyneat.Model` instances and runners for both MPKs.
5. Two MLA executions per denoising step.
6. A65 FP32 classifier-free guidance and Euler update.
7. One full-BF16 MLA vocoder invocation using the padded fixed latent; crop the
   time-major waveform output to `natural_latent_length * 3072` samples. Retain
   A65 ONNX Runtime as a fallback.
8. Sentence/clause chunking, followed by silence-aware joining or a short
   crossfade.

The MLA model must be loaded once, not once per branch, step, or utterance.
Input names, order, shapes, and dtypes must be checked against the artifact
manifest during worker startup.

## Remaining work

### 1. Hardware validation

- Load the tarball through public `pyneat` APIs.
- Confirm the MPK-reported contract and one-stage MLA execution.
- Run short and near-capacity deterministic tensors.
- Compare hardware output with saved AFE and ONNX outputs.
- Record warm and cold latency, transfers, and peak memory.

### 2. Persistent application worker

- Load one model and one runner during startup.
- Preallocate all fixed-size input/output buffers.
- Validate binary masks and select RoPE rows by mask length.
- Execute conditional and unconditional branches for all eight steps.
- Return the natural-length latent plus timing and health information.
- Recover cleanly from malformed requests and runner failures.

### 3. Vocoder MLA compilation

Target measurements show the 25.3M-parameter A65 vocoder is the dominant CPU
stage. The prepared graph compiles as one MLA plugin with only FP32 boundary
casts on EV74 and no A65 partition. INT8 weights are rejected because the
output-head PReLU amplifies their error: cropped cosine is only
`0.420`–`0.444`. Full BF16 raises cropped cosine to `0.992`–`0.996` and reduces
relative L2 from `0.897`–`0.909` to `0.093`–`0.125`. Hardware latency and
listening validation remain required; the compiler cycle estimate is
`11,625,345`.

### 4. Live-assistant chunking

- Prefer sentence boundaries.
- Split unusually long sentences at clause punctuation.
- Reject or split any chunk exceeding either 192-position limit.
- Target roughly 3–8 seconds of generated audio per chunk.
- Generate the next chunk while the current chunk is playing.
- Preserve the same voice/style tensors across chunks.

### 5. Artifact publication

Update `florianvoss/supertonic-3-sima` with:

- `supertonic_vector_field_sima_mpk.tar.gz`
- `supertonic_runtime_data.npz`
- the exact input contract and host-loop pseudocode
- compiler and upstream provenance
- the production validation report
- optional ONNX/quantized listening samples
- `supertonic_vocoder_sima_bf16_mpk.tar.gz`
- `vocoder_bf16_manifest.json` and the BF16-versus-ONNX validation report

Do not upload compiler-debug `.mlc` files. The ELF and MPK JSON are already
contained in the tarball.
