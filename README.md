# Supertonic 3 on SiMa Modalix

Hybrid text-to-speech for a SiMa Modalix DevKit. The duration predictor and
text encoder run once per utterance in persistent ONNX Runtime sessions. The
eight-step vector field and fixed-profile vocoder use persistent PyNeat model
runners on MLA. The host performs classifier-free guidance, Euler updates,
and WAV serialization.

The upstream model is pinned to revision
`724fb5abbf5502583fb520898d45929e62f02c0b`.

## Repository layout

```text
compilation/
  reference_cases.py      deterministic end-to-end validation fixtures
  vector_estimator/
    graph_surgery.py       public surgery pipeline
    transforms.py          private transformation implementation
    contract.py            fixed source and compiled tensor contracts
    compile.py             vector-estimator MPK packaging
  vocoder/
    graph_surgery.py       complete vocoder surgery pipeline
    compile.py             vocoder MPK packaging
app/
  supertonic_sima/        persistent hybrid runtime
  examples/
    simple.py             WAV-producing CLI with latency and RTF
    gui.py                browser voice/language playground
    speech_server.py      POST /v1/speech
scripts/
  setup_devkit.sh
requirements-devkit.txt
```

Generated models, compiler output, reports, and audio belong under ignored
workspace paths such as `build/`, `models/`, or `/media/nvme/supertonic-tts`.

## DevKit setup

Run this repository on the DevKit. The script creates an isolated virtual
environment under `/media/nvme/supertonic-tts`, downloads the matching PyNeat wheel with
`sima-cli neat install core -t pyneat`, installs ONNX Runtime and PyNeat into
that environment, and downloads the upstream CPU models plus compiled MLA
models. An existing application environment is recreated to keep the install
isolated and reproducible.

```bash
bash scripts/setup_devkit.sh
source /media/nvme/supertonic-tts/.venv/bin/activate
```

Override `SUPERTONIC_APP_ROOT` or `SUPERTONIC_REPO_ROOT` when the local DevKit
paths differ. The DevKit must have `sima-cli` installed and configured to access
the Neat artifacts.

## Examples

Generate a WAV. One warmup runs after constructing the sessions/runners; all
measured runs reuse those objects.

```bash
python app/examples/simple.py \
  --text "Hello from Modalix." --voice M1 --lang en \
  --output /media/nvme/supertonic-tts/output/hello.wav
```

The CLI prints audio length, generation time, real-time factor, latent length,
and per-stage latency. The default is eight denoising steps; use `--steps 12`
to compare the higher-quality schedule, `--runs N` for repeated warm
measurements, or `--vocoder-backend cpu` to compare the upstream CPU vocoder.

Start the browser UI:

```bash
python app/examples/gui.py --host 0.0.0.0 --port 8080
```

The browser fills the 192-character processed-text envelope and splits long
text at `.`, `!`, `?`, `:`, and `;` boundaries, including their Japanese/CJK
forms such as `。`, `！`, `？`, and `…`. An oversized segment falls back to
commas, then whitespace or a hard boundary. It synthesizes chunks sequentially
and starts playback as soon as the first chunk is ready while subsequent chunks
are synthesized into the continuous Web Audio queue. Inter-chunk outputs have
their generated trailing silence removed with a short retained tail and fade;
the final chunk remains unchanged.

The server logs each chunk's complete JSON-escaped text together with its
position and split boundary, raw/processed lengths, synthesis settings, latent
frames, audio duration, generation time, and RTF. The browser console
additionally reports the complete chunk plan and per-chunk trimming measurements.

Start the speech endpoint:

```bash
python app/examples/speech_server.py --host 0.0.0.0 --port 8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Guten Tag aus dem Modalix DevKit.","voice":"F1","language":"de"}' \
  --output speech.wav
```

Only WAV output is supported. Response headers include
`X-Audio-Length-Seconds`, `X-Generation-Length-Seconds`,
`X-Real-Time-Factor`, and `X-Latent-Length`. Requests are serialized around
the persistent MLA runners.

## Compiled contracts

The vector estimator accepts eight contiguous FP32 tensors in this order:

```text
noisy_latent       [1,144,1,192]
text_emb           [1,256,1,192]
style_ttl          [1,256,1,50]
style_key          [1,256,1,50]
latent_mask        [1,1,1,192]
text_mask          [1,192,1,1]
time_sinusoidal    [1,64,1,1]
rope_tables        [1,128,1,192]
```

It emits FP32 `velocity [1,144,1,192]`. The vocoder accepts FP32
`latent [1,144,1,192]` and emits FP32 `wav_frames [1,512,1,1152]`; public HWC
order is already time-major, so the runtime crops a flat view to the natural
predicted length.

The host creates the denoising timestep embeddings in memory for the selected
`--steps` value. Runtime data stores only the RoPE bank and CFG/style constants;
the app remains compatible with older archives that also contain a timestep
table.

Processed text and predicted latent length must both be at most 192. The
latent profile represents about 13.37 seconds at 44.1 kHz.

## Graph surgery and compilation

Keep the graph and released-package environments separate, then generate the
deterministic upstream reference cases:

```bash
python3 -m venv .venv-graph
.venv-graph/bin/pip install -r compilation/requirements.txt
python3 -m venv .venv-reference
.venv-reference/bin/pip install supertonic==1.3.1
.venv-reference/bin/python compilation/reference_cases.py \
  --model-dir models/supertonic-3 \
  --output-dir build/reference-cases \
  --summary build/reference-cases.json
```

The vector transformation validates those NPZs and produces the externalized
runtime table:

```bash
.venv-graph/bin/python compilation/vector_estimator/graph_surgery.py \
  --source models/supertonic-3/onnx/vector_estimator.onnx \
  --reference-dir build/reference-cases \
  --output-dir build/vector-surgery
```

The vocoder transformation accepts repeatable reference cases for optional
waveform equivalence validation:

```bash
.venv-graph/bin/python compilation/vocoder/graph_surgery.py \
  --input models/supertonic-3/onnx/vocoder.onnx \
  --output build/vocoder-surgery/supertonic_vocoder_sima.onnx \
  --report build/vocoder-surgery/report.json \
  --reference-case build/reference-cases/en_m1_short.npz
```

After quantizing each graph with the SiMa model compiler, package the saved
single-MLA network with the component command:

```bash
python compilation/vector_estimator/compile.py \
  --network-directory build/vector-quantized/supertonic_vector_field_sima \
  --output-directory build/vector-compiled \
  --compiler-debug-directory build/vector-compiler-debug

python compilation/vocoder/compile.py \
  --network-directory build/vocoder-quantized/supertonic_vocoder_sima \
  --output-directory build/vocoder-compiled \
  --compiler-debug-directory build/vocoder-compiler-debug
```

The vector profile uses BF16 activations and symmetric per-channel INT8
weights. The vocoder must use BF16 activations and BF16 weights; INT8 weights
damage its quantization-sensitive output head.
