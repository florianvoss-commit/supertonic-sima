# Supertonic 3 BF16 Vector Estimator on Modalix

## Objective

Replace Neat GenAI Studio's CPU Piper synthesis path with Supertonic 3 while
offloading only Supertonic's denoising/vector-estimator model to Modalix MLA.
Use one compiled profile:

```text
batch_size   = 1
text_length  = 192
latent_length = 192
precision    = BF16 activations + BF16 weights
```

The duration predictor, text encoder, text processing, voice-style loading, and
vocoder remain CPU ONNX Runtime components in the first implementation. The host
keeps the denoising loop and invokes the same persistent MLA runner once per
denoising step.

This plan deliberately does not add a 320 profile. Text that does not fit the
192 profile must be split; it must never be silently truncated.

## Known baseline

- Source model: `supertonic/vector_estimator.onnx`
- Source SHA-256:
  `883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c`
- Hugging Face model: `Supertone/supertonic-3`
- Pinned model revision:
  `724fb5abbf5502583fb520898d45929e62f02c0b`
- Python package used during investigation: `supertonic==1.3.1`
- Vector-estimator input and output dtype: FP32 at the ONNX/public runtime
  boundary; the compiled model performs BF16 computation internally.
- One latent position represents `512 * 6 = 3072` waveform samples, or
  approximately `69.6599 ms` at 44.1 kHz.
- A 192-position latent therefore represents at most approximately 13.37 seconds
  of audio.

Measured M1 English shapes at speed 1.0:

| Case | Raw characters | Preprocessed text length | Predicted latent length | Predicted audio |
|---|---:|---:|---:|---:|
| Short benchmark | 51 | 60 | 60 | 4.14 s |
| Medium benchmark | 143 | 152 | 149 | 10.32 s |

The 192 profile fits both benchmark cases. It does not guarantee that every
string shorter than 192 characters fits: Unicode normalization can expand text,
and the duration predictor, voice, language, punctuation, and speed independently
control latent length.

## Fixed model contract

The compiled vector-field core must expose the following ordered, all-4D tensor
contract. Classifier-free guidance is no longer encoded as an internal batch of
two. The worker invokes this batch-1 core once for the conditional context and
once for the unconditional context, then applies the original guidance and
Euler update on the host.

| Index | Name | Shape | Boundary dtype |
|---:|---|---|---|
| 0 | `noisy_latent` | `[1, 144, 1, 192]` | FP32 |
| 1 | `text_emb` | `[1, 256, 1, 192]` | FP32 |
| 2 | `style_ttl` | `[1, 256, 1, 50]` | FP32 |
| 3 | `style_key` | `[1, 256, 1, 50]` | FP32 |
| 4 | `latent_mask` | `[1, 1, 1, 192]` | FP32 |
| 5 | `text_mask` | `[1, 192, 1, 1]` | FP32 |
| 6 | `current_step` | `[1, 1, 1, 1]` | FP32 |
| 7 | `total_step` | `[1, 1, 1, 1]` | FP32 |
| output | `velocity` | `[1, 144, 1, 192]` | FP32 |

The implementation must query and record the compiled package's input specs and
must fail startup if names, order, shapes, or dtypes differ from this manifest.
Do not infer input order from dictionary iteration at runtime.

## Planned repository changes

### Model preparation and compiler tools

Add these files under `supertonic/`:

```text
supertonic/
  IMPLEMENTATION_PLAN.md
  vector_estimator.onnx
  tools/
    staticize_vector_estimator.py
    generate_reference_cases.py
    verify_onnx_equivalence.py
    compile_vector_estimator.py
    verify_compiler_outputs.py
    benchmark_modalix.py
  testdata/
    manifest.json
    reference_cases/             # generated locally; large arrays not committed
  build/                         # generated; not committed
```

The tools must write machine-readable JSON results, including model hashes,
compiler version, package version, shape profile, precision, numerical metrics,
and command-line arguments. Generated ONNX, quantized, compiled, NPZ, MPK, WAV,
and log artifacts belong under ignored `build/` or `testdata/reference_cases/`
paths unless repository policy explicitly chooses to publish them.

### GenAI Studio integration

Add or update the following application components:

```text
apps/examples/genai/neat-genai-studio/
  setup.sh
  run.sh
  README.md
  src/python/requirements.txt
  src/python/ui/flask_app.py
  src/python/ui/voice_catalog.py
  src/python/ui/supertonic_tts.py
  src/python/ui/supertonic_mla_worker.py
  src/python/ui/test_supertonic_tts.py
```

Responsibilities:

- `supertonic_tts.py` owns text processing, duration prediction, text encoding,
  shape admission, padding, calls to the MLA worker, cropping, CPU vocoding, WAV
  creation, and the Piper-compatible `synthesize`/`synthesize_stream` facade.
- `supertonic_mla_worker.py` runs under `PYNEAT_PYTHON`, loads the compiled model
  once, builds one persistent tensor-input runner, executes all requested
  denoising steps, and returns one padded latent. It must not load or initialize
  an MLA model per step or utterance.
- The UI process sends one request per utterance to the worker. The request
  contains the padded initial latent, text embedding, style, both masks, and the
  total step count. The worker performs the entire denoising loop so that IPC is
  not crossed eight times.
- IPC should reuse the existing worker principles: length-prefixed messages,
  bounded payload sizes, explicit protocol version, structured error replies,
  one-request lock, health handshake, clean shutdown, and restart after a worker
  crash. Use raw contiguous tensor bytes plus a small JSON header rather than
  compressed NPZ for the hot path.
- Keep the Studio UI virtual environment and the Neat/pyneat environment
  separate. Install `supertonic==1.3.1` into the UI environment. The MLA worker
  needs only the existing `pyneat`, NumPy, and standard library available to
  `PYNEAT_PYTHON`.
- Pin the Supertonic model revision in setup. Download the configuration,
  Unicode indexer, voice styles, duration predictor, text encoder, and vocoder.
  The CPU path should not load the 245 MB vector-estimator ONNX after the MLA
  path is validated.
- Add configuration/environment variables for the CPU model directory, compiled
  vector-estimator MPK, voice, steps, and CPU ORT thread count. Validate all
  configured paths before starting the worker.
- Initially expose Supertonic as a selectable engine. Make it the default and
  remove Piper from the required setup only after standalone and concurrent-LLM
  acceptance gates pass. Keeping Piper as a temporary fallback makes comparison
  and rollback possible.

## Phase 1: freeze the ONNX shape

**Status: Completed (2026-08-26, including the all-4D activation rewrite)**

### Initial static-shape baseline findings

- Added `tools/model_contract.py` and `tools/staticize_vector_estimator.py`.
  Staticization refuses an unrecognized source model, rewrites symbolic
  dimensions in inputs/outputs/value-info, runs full ONNX checking and strict
  shape inference, validates the ordered FP32 contract, and writes a JSON
  provenance report.
- Verified source SHA-256:
  `883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c`.
- Produced the shape-only artifact at
  `build/onnx/vector_estimator_b1_t192_l192.static.onnx`, SHA-256
  `062c0a3680bd46e640b80dbfe0a67dfaa8d6d4d41e032855961199e45e44d917`.
- Frozen 22 occurrences of `batch_size`, 178 occurrences of `text_length`, and
  27 occurrences of `latent_length`. The graph remains 1,004 nodes and 446
  initializers at ONNX opset 19; no mathematical nodes or names were changed.
- `onnx.checker.check_model(..., full_check=True)`, strict shape inference, an
  artifact reload, and ONNX Runtime session creation all passed. ONNX Runtime
  reports exactly the seven inputs and one output specified by the fixed model
  contract.
- The release-2.1 BF16 audit inspected all 1,004 nodes and reported 18 database
  matches, one stale `Softplus` rejection, and 13 operators absent from that
  database. As established with the current compiler support information,
  `LayerNormalization`, `Sin`, `Cos`, `Softplus`, and Conv1D lowering are
  supported; the audit JSON is retained as informational evidence at
  `build/reports/phase1_bfloat16_audit.json`.
- No model surgery was justified or performed. The optional `onnxsim` artifact
  was not produced in this phase: Python 3.12 had no matching wheel and the SDK
  shell's ARM cross-compiler environment cannot build a host extension. The
  validated shape-only artifact is the Phase 2/3 reference and the first Phase 4
  compiler input. Simplification remains an optional compile-failure fallback,
  not a Phase 1 gate.
- Machine-readable staticization evidence is in
  `build/reports/phase1_staticize.json`; console output is retained in
  `build/phase1_staticize.log`.

These findings establish the rank-preserving static baseline, not the final
Phase 1 artifact. Phase 1 was reopened after the compiler-facing requirement
changed: all public inputs, the output, and every data activation must be rank 4
with a leading batch dimension of 1. Shape/index tensors and scalar
initializers are not data activations and may remain lower rank. The rewrite
also removes rank-adaptation `Unsqueeze`/`Squeeze` nodes wherever the 4D
representation makes them redundant.

### All-4D surgery findings

- Added `tools/extract_vector_field.py`. It removes the released model's
  internal two-element classifier-free-guidance batch and Euler wrapper, and
  exposes one true batch-1 vector-field evaluation. `style_key` is explicit so
  the same compiled core can execute both conditional and unconditional
  branches. The worker must compute `4 * conditional - 3 * unconditional`,
  divide by `total_step`, add to the current latent, and apply `latent_mask`.
- The exact rank-3 extraction is
  `build/onnx/vector_field_b1_t192_l192.rank3.onnx`, SHA-256
  `b0ad3521b2190a81f039dd3bf25bfc38704218b856f66427f3ceee6665220e62`.
  Both extracted branches match the corresponding raw released-model branches:
  maximum absolute errors were `1.9073486e-06` (conditional) and
  `5.722046e-06` (unconditional).
- Added `tools/lift_vector_field_to_4d.py`. Channel-first activations use
  `[B,C,1,L]`, channel-last contexts use `[B,1,L,C]`, and attention uses
  `[B,H,L,D]`. It converts all 86 Conv1D nodes and weights to Conv2D, normalizes
  head-first attention to batch-first, replaces rank adaptation with permanent
  4D layouts, folds static shape chains, and prunes 52 dead nodes.
- The final compiler artifact is
  `build/onnx/vector_field_b1_t192_l192.all4d.opt.onnx`, SHA-256
  `00d9562f94450822cc862ed358b179531622e433fbdd072dc6f9ef4a8ee41a51`.
  It contains 933 nodes. It has zero `Gather`, zero
  `Squeeze`, zero `Unsqueeze`, zero `Equal`, and zero `Where` nodes. Pre-softmax
  binary key masking uses the exact arithmetic identity
  `scores + (1 - reciprocal(mask))`; post-softmax query masking uses direct
  multiplication. All eight inputs, the output, and every live data activation
  are rank 4 with leading batch 1. The only lower-rank live tensors are two
  immutable INT64 Slice-index constants, which are operator parameters rather
  than activations.
- Full ONNX checking, strict shape inference, model reload, and ONNX Runtime
  execution pass. The all-4D core was bit-exact (`max_abs=0`) against the
  extracted batch-1 core for two random masked cases. Reconstructing the full
  two-pass guidance and Euler update matched the released model with maximum
  absolute error `2.3841858e-06`.
- Machine-readable extraction and rank-lift evidence is in
  `build/reports/phase1_extract_vector_field.json` and
  `build/reports/phase1_lift_all4d.json`.
- Added `tools/optimize_vector_field_4d.py`. Following the optimized attention
  implementation in `llima/sima_lmm/model/language_cache_model.py`, all eight
  attention blocks now keep Q/K/V as `[B,D,H,L]`, compute scores and values with
  the two LLiMa `Einsum` equations, reduce Softmax over key length, and merge
  heads with Split/Concat. Linear attention projections are 1x1 Conv2D. The
  time MLP and its four stage projections are also channel-first Conv2D.
- The final graph contains exactly 72 `Transpose` nodes: two required layout
  adapters around each of 36 channel-only `LayerNormalization` nodes. There are
  no attention, mask, time-encoder, identity, or reshape transposes remaining.
  There are also zero `Identity` and zero `Reshape` nodes.
- Zero-padding the released dynamic model to 192 was proven invalid because its
  28 edge-padded ConvNeXt convolutions use the physical tensor endpoint. The
  optimized graph computes the final valid activation from `latent_mask` and
  fills the invalid tail before every edge Pad. This restores natural-length
  boundary behavior without Gather/Squeeze/Unsqueeze/Equal/Where and keeps all
  floating activations rank 4 with leading batch 1.
- `onnx.checker`, strict shape inference, ONNX Runtime loading, forbidden-op
  checks, and the all-rank4 audit pass. The final machine-readable surgery and
  equivalence evidence is in `build/reports/phase1_optimize_all4d.json` and
  `build/reports/phase1_optimize_equivalence.json`.

### Staticization

Create a shape-only static model first. Replace every symbolic occurrence of:

```text
batch_size=1
text_length=192
latent_length=192
```

Expected output:

```text
supertonic/build/onnx/vector_estimator_b1_t192_l192.static.onnx
```

The staticization tool must:

1. Verify the source SHA-256 before editing.
2. Rewrite graph inputs, outputs, and value-info symbolic dimensions.
3. Run `onnx.checker.check_model` and ONNX shape inference.
4. Print and persist the final ordered input/output contract.
5. Preserve all tensor and node names unless simplification explicitly removes
   internal nodes.

Do not combine shape rewriting and graph simplification in the first artifact.
After the shape-only model passes equivalence, optionally create:

```text
supertonic/build/onnx/vector_estimator_b1_t192_l192.sim.onnx
```

with `onnxsim` and fixed overwrite shapes. Use the simplified model only if it
also passes the complete equivalence suite. This makes it possible to distinguish
a shape-metadata problem from a simplifier rewrite problem.

### Operator audit

Run the Modalix BF16 operator audit and retain its JSON report, but treat the
current compiler import/compile result as authoritative. The release-2.1 audit
database is known not to represent current support for `LayerNormalization`,
`Sin`, `Cos`, and `Softplus`. Conv1D is expected to lower to Conv2D. Audit warnings
for those cases are informational and not a reason to rewrite model math.

Do not perform model surgery without a concrete importer, quantizer, or compiler
failure and a minimal reproducer.

## Phase 2: generate representative inputs

**Status: Completed (2026-08-26)**

### Findings

- Added deterministic generation and hash-verified resume support in
  `tools/generate_reference_cases.py`.
- Generated and hash-verified five accepted cases: English M1 short and medium,
  English F1 short, Spanish M1 short, and an English M1 near-capacity case.
  Natural `(T,L)` values include `(60,60)`, `(152,149)`, `(83,88)`, and the
  near-capacity case `(186,182)`.
- Stored natural and padded inputs/masks, every one of eight denoising outputs,
  final cropped latent, waveform, metadata, seeds, model hashes, and WAV/NPZ
  hashes beneath `testdata/reference_cases/`; `testdata/manifest.json` verifies
  all five artifact pairs without a hash error.
- Boundary evidence confirms text length 192 and latent length 192 are admitted,
  193 is rejected for either dimension, the medium prompt at speed 0.5 predicts
  latent length 297 and is rejected, and empty/private-use input is rejected.

Generate deterministic reference cases using the pinned official Supertonic CPU
components. Store the random seed and every model input, intermediate, and output
needed to replay a case.

Required cases:

- English M1 short benchmark, steps 1, 2, and 8.
- English M1 medium benchmark, steps 1, 2, and 8.
- English F1 short case to exercise a different style vector.
- One non-English supported-language case.
- A near-capacity case with both post-padding lengths at most 192 and at least
  one natural length greater than or equal to 176.
- Boundary/error cases: text length exactly 192, text length 193, latent length
  exactly 192, latent length 193, speed 0.5, empty text, and unsupported input
  characters.

For each accepted case, record:

- raw and normalized text;
- language, voice, speed, total steps, and random seed;
- natural `T` and `L`;
- unpadded CPU inputs;
- padded 192 inputs and masks;
- output after every denoising step;
- cropped final latent;
- CPU-vocoder waveform and sample count.

Generate the natural-size random latent first and then pad it. Generating a
192-wide random tensor directly changes NumPy's per-channel random sequence and
would make deterministic comparisons unnecessarily difficult.

## Phase 3: prove ONNX numerical equivalence

**Status: Completed (2026-08-26)**

### Findings

- Added `tools/verify_onnx_equivalence.py` and retained the complete metrics in
  `build/reports/phase3_onnx_equivalence.json`.
- Released dynamic versus shape-only static execution passes the original strict
  gate for all five cases.
- The required diagnostic demonstrated that naively zero-padding the released
  graph is not invariant. After the mask-aware edge-boundary rewrite, fixed-192
  execution passes padding/masking invariance for all five cases and produces a
  zero padded tail.
- Actual conditional and unconditional special tokens were read from the pinned
  released graph. Teacher-forced and free-running 1/2/8-step comparisons pass
  the surgery gate for both benchmark prompts. The optimized Einsum/Conv kernels
  change FP32 accumulation order: worst eight-step free-running error is
  `max_abs=1.8697977e-4`, `relative_l2=2.0516860e-5`, and
  `cosine=0.9999999997895405`. The report preserves the original stricter
  elementwise result as a failed diagnostic rather than claiming bit identity.

Three separate checks are required.

### A. Shape rewrite equivalence

Run the dynamic source model and the shape-only static model on the exact same
padded 192 inputs. Compare the complete `[1, 144, 192]` output.

Pass criteria:

- output shape and dtype are identical;
- no NaN or infinity;
- `numpy.allclose(reference, candidate, rtol=1e-5, atol=1e-6)`;
- relative L2 error at most `1e-5`;
- cosine similarity at least `0.999999`.

The same criteria apply between the shape-only and simplified ONNX artifacts.

### B. Padding and masking invariance

Run the original dynamic model twice: once at its natural `T`/`L`, and once with
the same tensors padded to 192 and the supplied masks set to zero in padded
positions. Crop the padded output back to natural `L` before comparison.

Pass criteria for valid positions:

- `numpy.allclose(reference, cropped_padded, rtol=1e-4, atol=1e-5)`;
- relative L2 error at most `1e-4`;
- cosine similarity at least `0.99999`.

Also report the padded-tail maximum absolute value. Do not assume the tail is
zero until this test demonstrates it. The CPU vocoder must always receive the
cropped natural-length latent, never the padded tail.

### C. Iterative denoising equivalence

For steps 1, 2, and 8, run two comparisons:

- **Teacher-forced:** give CPU ONNX and the candidate model the same FP32 latent
  at each step. This isolates one invocation's error.
- **Free-running:** feed each implementation's own result into its next step.
  This measures accumulated denoising error.

The shape-only/static ONNX chain must meet the strict criteria from check A at
every step. Save per-step metrics rather than reporting only the final tensor.

## Phase 4: compile BF16 for Modalix

**Status: Not started**

### Compiler environment

Use an activated SiMa Model Compiler environment containing `afe`, `onnx`,
`onnxsim`, NumPy, and the Modalix target definitions. Record the compiler and SDK
versions in the build manifest.

The generic image quantize/compile helper must not be used unchanged: it assumes
image-shaped inputs and transposes NCHW tensors to NHWC. Supertonic has seven
heterogeneous rank-1/rank-3 inputs. Implement `compile_vector_estimator.py` as a
small model-specific driver using the same supported compiler APIs.

### Compiler driver behavior

The driver must:

1. Import the static ONNX with `gen2_target` and the exact seven-name, seven-shape
   contract listed above.
2. Keep the public/import boundary FP32.
3. Select `bfloat16_scheme()` for both activations and weights.
4. Feed representative Supertonic NPZ tensors, not random image data, to any
   compiler API that requires an example/calibration iterable.
5. Execute the loaded FP model and BF16 quantized model on the same cases before
   compiling.
6. Compile with batch size 1.
7. Package the output as a Neat-loadable compiled model archive.
8. Write a manifest containing source and static ONNX hashes, HF revision,
   package version, compiler version, precision, input order/shapes/dtypes,
   output contract, and build timestamp.

Expected invocation after the tools exist:

```bash
activate-model-compiler

python supertonic/tools/compile_vector_estimator.py \
  --model supertonic/build/onnx/vector_estimator_b1_t192_l192.sim.onnx \
  --reference-dir supertonic/testdata/reference_cases \
  --build-dir supertonic/build/modalix-bf16-b1-t192-l192 \
  --device modalix \
  --bf16-weights \
  --bf16-activations
```

If the simplified artifact fails import for a simplifier-specific reason, retry
the already-validated shape-only static artifact. If compilation exhausts host
memory, lower `SIMA_MLA_SIM_PARALLEL` incrementally and rerun the identical
command; do not default directly to one simulator worker.

### Pre-compile BF16 numerical gate

Compare Model Compiler FP execution and BF16 quantized execution with the same
representative tensors.

Initial acceptance thresholds:

- no NaN or infinity;
- teacher-forced per-step cosine similarity at least `0.995`;
- teacher-forced per-step relative L2 error at most `0.05`;
- free-running eight-step final cosine similarity at least `0.99`;
- free-running eight-step final relative L2 error at most `0.10`.

These thresholds are gates, not proof of audio quality. If they fail, inspect
per-layer error and listen to generated audio before changing precision or graph
math. Any threshold adjustment must be documented with the measured distributions
and an audio-quality justification.

## Phase 5: build the Modalix runner

**Status: Not started**

Use the public `pyneat.Model` tensor-input route because the caller already owns
all seven decoded tensors.

Worker startup sequence:

1. Verify the compiled artifact and manifest exist.
2. Construct `pyneat.ModelOptions` with tensor input and no image preprocessing.
3. Load `pyneat.Model` once.
4. Inspect `input_specs()`, `output_specs()`, and model metadata.
5. Validate them against the checked-in contract/compiled manifest.
6. Create seven contiguous FP32 seed tensors at the fixed shapes.
7. Build one persistent model runner.
8. Complete a one-step warm-up and health handshake before reporting ready.

For every request:

1. Validate protocol version, shapes, dtypes, finite values, masks, and
   `1 <= total_steps <= 15`.
2. Set `current_step=[step]` and `total_step=[total_steps]` as FP32.
3. Submit the seven tensors in manifest order.
4. Check push/run success, timeout, output count, output shape, dtype, and finite
   values.
5. Reuse the returned denoised latent as the next step's `noisy_latent`.
6. Return the final padded FP32 latent and per-step timing.

The worker must close the runner and model cleanly on shutdown and on SIGTERM.
Malformed input must produce a structured error without terminating the worker.

## Phase 6: implement fixed-shape admission and synthesis

**Status: Not started**

For each sanitized Studio TTS segment:

1. Apply the pinned Supertonic Unicode/language preprocessing.
2. If preprocessed `T > 192`, split the raw segment at punctuation or whitespace
   and recursively process both parts.
3. Run the CPU duration predictor at the selected speed.
4. Calculate
   `L = ceil(duration_seconds * 44100 / 3072)` using the same integer behavior as
   the official SDK.
5. If `L > 192`, split and rerun preprocessing and duration prediction. Do not
   clamp duration or crop speech.
6. Run the CPU text encoder at natural `T`.
7. Generate the initial random latent at natural `[1, 144, L]`.
8. Right-pad `text_emb`, `text_mask`, `noisy_latent`, and `latent_mask` to their
   fixed 192 dimensions. Preserve `style_ttl` as `[1, 50, 256]`.
9. Send one request to the MLA worker for all denoising steps.
10. Crop the returned latent to natural `L`.
11. Run the dynamic CPU vocoder on the cropped latent.
12. Trim or validate the waveform against the predicted duration using the same
    policy as the pinned official SDK, then create 44.1 kHz mono FP32/PCM WAV.

Splitting policy requirements:

- Prefer sentence punctuation, then comma/semicolon/colon, then whitespace.
- Search around the midpoint to avoid pathological one-character fragments.
- Guarantee progress for text with no whitespace.
- Preserve text order and punctuation.
- Run shape admission again after every split because character count alone is
  not the model contract.
- At speed 0.5, expect more duration-driven splits; do not add a hidden 384/640
  profile.

`synthesize_stream` should yield one WAV buffer per admitted Supertonic segment.
This provides inter-segment streaming, although Supertonic still does not stream
within a single denoised segment.

## Phase 7: verify compiled execution on Modalix

**Status: Not started**

Deploy the MPK and test tools beneath `/workspace` or `/media/nvme`. Prefer the
standard DevKit runner when available; direct SSH is an acceptable fallback only
when `dk`/`devkit-run` is unavailable.

### Standalone tensor verification

Replay every reference case through the persistent `pyneat.Model` runner.
Compare the hardware result to CPU ONNX using both teacher-forced and free-running
chains.

Use the BF16 thresholds from Phase 4 and save:

- per-step latency;
- max/mean absolute error;
- relative L2 error;
- cosine similarity;
- NaN/Inf count;
- model load and warm-up time;
- peak process RSS and available CMA before/after load and after shutdown.

Verify that repeated load/run/shutdown cycles return CMA allocations to baseline.

### End-to-end waveform verification

Feed the CPU vocoder first with the CPU-reference final latent and then with the
Modalix final latent.

Required checks:

- identical waveform rank, channel count, sample rate, and expected sample-count
  policy;
- no clipping beyond the reference policy, NaN, infinity, or long non-speech tail;
- waveform cosine similarity at least `0.99` on aligned samples;
- log-mel cosine similarity at least `0.99`;
- manual A/B listening for pronunciation, speaker identity, noise, clicks,
  truncation, repeated syllables, and end-of-utterance artifacts.

Waveform metrics are diagnostic rather than sufficient: an eight-step BF16 chain
can have benign sample-level differences while remaining perceptually equivalent.
The listening gate is mandatory before changing the Studio default.

## Phase 8: test Neat GenAI Studio integration

**Status: Not started**

### Unit tests

Run host/unit tests for:

- exact padding shapes and mask values;
- natural length 192 accepted and 193 split;
- duration-driven split at latent length 193;
- recursive splitting with punctuation, whitespace, and no whitespace;
- speed 0.5 behavior;
- deterministic random seed and natural-latent-then-pad behavior;
- output crop before vocoding;
- IPC framing, partial reads, size limits, malformed headers, worker error,
  timeout, crash, restart, and shutdown;
- compiled-manifest mismatch for name, order, dtype, or shape;
- stale generation output discarded after Studio cancellation;
- WAV format and duration accounting;
- language and voice routing.

Run the existing Studio TTS sanitization, voice catalog, ASR metadata, and UI
tests to catch regressions outside the new engine.

### Standalone application smoke test

With no LLM loaded:

1. Start Studio and select Supertonic M1.
2. Synthesize short and medium benchmark text.
3. Exercise English plus one other supported language.
4. Exercise speed 0.5, 1.0, and 2.0.
5. Exercise a segment requiring text-driven splitting and one requiring
   duration-driven splitting.
6. Cancel a response during synthesis and confirm stale audio is not emitted.
7. Kill the MLA worker and confirm the client reports/restarts it without taking
   down the UI.

### Shared-MLA/concurrent-LLM gate

This is a required architecture gate. Today LLiMa may access `mla-rt` directly
while a classic Neat model runs through the dispatcher/`mlashmcomplex` path.
Studio begins TTS while LLM token generation may still be active, so standalone
TTS success does not prove the two paths can safely coexist.

Test all of the following with a chat model loaded:

- LLM generation alone;
- Supertonic inference alone while the LLM model remains resident;
- TTS started after LLM generation is completely idle;
- TTS started while tokens are still being generated;
- repeated alternating and overlapping requests;
- cancellation and process shutdown during overlap.

Collect LLiMa, dispatcher, `mlashmcomplex`, kernel/driver, and Studio logs. Check
for deadlock, model corruption, timeout, daemon restart, stale allocation, or
incorrect output.

Do not enable Supertonic as Studio's default if overlapping access is unsafe.
The preferred long-term resolution is one shared MLA gatekeeper. A temporary
fallback may defer TTS until LLM generation is idle, but that changes current
streaming behavior and must be explicit in the UI and documentation.

### Performance benchmark

Use the same texts, M1 voice, English, speed 1.0, and denoising-step count as the
CPU investigation. Perform one untimed warm-up followed by exactly one measured
short run and one measured medium run.

Report:

- model initialization and warm-up time;
- duration/text encoder, MLA denoising total and per-step, vocoder, IPC, and total
  latency;
- generated audio duration;
- RTF = total wall time / generated audio duration;
- time to first emitted WAV segment;
- peak RSS and CMA usage;
- comparison with the recorded Piper and Supertonic CPU baselines.

The hard performance gate for replacement is RTF below 1.0 for both cases at the
chosen production step count. Also report whether time to first audio is better
or worse than Piper; do not hide the fact that Supertonic emits only after one
admitted segment completes.

## Acceptance criteria

The implementation is complete only when all of the following are true:

- The source model hash and pinned upstream revision are recorded and verified.
- The shape-only and any simplified ONNX pass checker, shape inference, and the
  strict ONNX equivalence gates.
- Padding/masking invariance is demonstrated on every accepted reference case.
- The BF16 compiler simulation and Modalix hardware meet the documented tensor
  accuracy gates for teacher-forced and free-running 1/2/8-step execution.
- The compiled artifact loads through public `pyneat.Model`, exposes the exact
  seven-input contract, and uses one persistent runner.
- Text or latent lengths above 192 are split without truncation.
- CPU vocoding always receives a natural-length cropped latent.
- End-to-end WAVs pass structural, numerical, and listening checks.
- Unit, Studio regression, standalone DevKit, cleanup, and worker-recovery tests
  pass.
- The shared-MLA/concurrent-LLM gate passes, or Studio explicitly serializes TTS
  after LLM inference and documents the behavior.
- One-run short and medium Modalix benchmarks are recorded, and both production
  RTF values are below 1.0.
- Setup, configuration, model licensing/attribution, rollback, and troubleshooting
  instructions are documented in the Studio README.

## Execution order

1. Build deterministic reference cases from the pinned CPU implementation.
2. Staticize to 192 and prove shape and padding equivalence.
3. Implement the model-specific BF16 compiler driver and simulator comparison.
4. Compile and package the Modalix model.
5. Build the standalone persistent pyneat worker and verify hardware tensors.
6. Verify end-to-end audio with the CPU vocoder.
7. Add fixed-shape admission, splitting, and the Studio-compatible TTS facade.
8. Add setup/configuration and selectable-engine integration.
9. Run standalone Studio and performance tests.
10. Run the shared-MLA/concurrent-LLM gate.
11. Make Supertonic the default only after all acceptance gates pass.
