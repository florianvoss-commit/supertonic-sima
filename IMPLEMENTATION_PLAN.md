# Supertonic 3 Modalix deployment plan

## Goal

Run Supertonic 3 speech synthesis with its iterative vector field on Modalix
MLA and the remaining pipeline on the embedded A65. Use sentence-sized chunks
and a single fixed batch-one, 192-position MLA profile.

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
| Run compiled MPK on Modalix hardware | Pending |
| Integrate a persistent `pyneat` worker | Pending |
| Publish refreshed Hugging Face artifacts | Pending |

## Fixed profile and interface

- Batch: `1`
- Maximum processed text length: `192`
- Maximum latent length: `192`
- Maximum audio duration: approximately `13.37 s`
- Boundary dtype: FP32
- Activation precision: BF16
- Weight precision: symmetric per-channel INT8
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
4. A persistent `pyneat.Model` and runner for the MPK.
5. Two MLA executions per denoising step.
6. A65 FP32 classifier-free guidance and Euler update.
7. A65 ONNX Runtime vocoding of the cropped natural-length latent.
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

### 3. A65 pipeline benchmark

The vector estimator was about 90% of CPU model time in the recorded x86
profile. After MLA offload, benchmark the 25.3M-parameter vocoder on the actual
A65. Compile the vocoder for MLA only if it prevents real-time streaming.

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

Do not upload compiler-debug `.mlc` files. The ELF and MPK JSON are already
contained in the tarball.
