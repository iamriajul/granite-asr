# granite-asr

OpenAI-compatible speech-to-text service wrapping
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) with IBM
**Granite Speech 5.0 470M TurboCTC** on the Vulkan/ggml backend.

Single static Go binary — no interpreter, no runtime dependencies.

## Why this exists

transcribe.cpp is the reference ggml runtime for Granite 5 TurboCTC and has
first-class Vulkan support. That combination is what makes the model usable on
AMD Polaris (gfx803, e.g. RX 570), where AMD dropped ROCm support after 3.5 and
modern stacks reject the card outright. Vulkan goes through Mesa RADV, which
still supports Polaris.

Upstream ships a CLI but no HTTP server. This is that server.

## Pipeline

```
upload → ffmpeg (16 kHz mono PCM) → transcribe.cpp → [optional polish] → JSON
```

Granite 5 TurboCTC is a CTC model: one forward pass, greedy decode, no
autoregressive token generation. On an RX 570 it runs at ~70× realtime
(114 ms for an 8 s clip, 203 ms for 15 s).

Its one drawback is **orthography**: CTC decoding emits normalised lowercase
with no punctuation. That is the expected raw output, not a bug.

## Polish stage (opt-in)

The optional polish pass restores casing and punctuation through an
OpenAI-compatible chat endpoint. It is **never applied unless a request asks
for it** — you always get the raw transcript verbatim unless you opt in.

Precedence, highest first:

1. explicit `polish=true|false` form field
2. server default (`ASR_LLM_DEFAULT_ON`, defaults to off)

The pass is instruction-constrained to add only orthography, never to alter
wording. **Any failure falls back to the raw transcript** — a missing or slow
LLM degrades output quality, never availability.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible (multipart `file=`) |
| `GET` | `/health` | liveness and model state |
| `GET` | `/v1/models` | model list (OpenAI shape) |

### Request fields

Standard OpenAI fields apply (`file`, `response_format`, `language`).
One extension:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `polish` | bool | server default | run the orthography pass |
| `response_format` | `json` \| `verbose_json` | `json` | `verbose_json` adds `raw_text`, `polished`, `duration`, `backend` |

With `response_format=verbose_json` you always receive `raw_text` alongside the
final `text`, so it is always clear what the polish stage changed.


## Container image

Published to `ghcr.io/iamriajul/granite-asr` on every push to `main`:

```bash
docker pull ghcr.io/iamriajul/granite-asr:main
```

The image contains the static service binary, `transcribe-cli` built with
`TRANSCRIBE_VULKAN=ON`, ffmpeg, and the Vulkan/Mesa user-space stack. The model
is **not** baked in — mount it from a volume.

Tags: `:main`, `:latest` (default branch), `:sha-<short>`, `:vX.Y.Z` for
releases.

## Deploying

```bash
kubectl apply -f k8s.yaml
```

Two host-specific details in that manifest are worth calling out, because both
fail *silently* rather than erroring:

- **GIDs.** The pod adds `render` (992) and `video` (44) as supplemental
  groups. Host-path device mounts do not carry host group ownership into the
  container, so without these RADV gets `EPERM` and falls back to llvmpipe —
  the service stays up and answers requests, just on the CPU. Check yours with
  `getent group render video`.
- **`RADV_PERFTEST=nogttspill`.** Without it Mesa intermittently spills VRAM
  into GTT on Polaris, leaving memory clocks low and destroying throughput.

Only one replica: the GPU is a single shared resource and the model is resident
in VRAM, so replicas would contend for it.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ASR_MODEL` | *(required)* | path to the GGUF model |
| `ASR_BACKEND` | `auto` | `vulkan` / `cpu` / `auto` |
| `ASR_THREADS` | `4` | CPU threads for decode |
| `ASR_BINARY` | `transcribe-cli` | transcribe.cpp binary to invoke |
| `PORT` | `8080` | listen port |
| `ASR_MAX_UPLOAD_MB` | `64` | upload ceiling |
| `ASR_MAX_SECONDS` | `120` | audio duration ceiling |
| `ASR_LLM_BASE_URL` | *(off)* | OpenAI-compatible base URL for polish |
| `ASR_LLM_MODEL` | *(off)* | model id for the polish pass |
| `ASR_LLM_API_KEY` | — | bearer token for the polish pass |
| `ASR_LLM_TIMEOUT` | `20` | seconds |
| `ASR_LLM_DEFAULT_ON` | `false` | polish every request unless overridden |

With no `ASR_LLM_*` set the service makes no outbound calls at all.

## Build

```bash
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o granite-asr .
```

Produces a ~7 MB static binary.

## Run

```bash
export ASR_MODEL=/models/granite5-NC-Q8_0.gguf
export ASR_BACKEND=vulkan
export RADV_PERFTEST=nogttspill   # prevents Mesa VRAM→GTT spill on Polaris
export ASR_BINARY=/path/to/transcribe-cli
exec ./granite-asr
```

`ffmpeg` must be on `PATH`.

## Usage

```bash
# raw transcript (default)
curl -s localhost:8080/v1/audio/transcriptions -F file=@clip.wav

# with orthography pass
curl -s localhost:8080/v1/audio/transcriptions -F file=@clip.wav -F polish=true

# inspect both forms
curl -s localhost:8080/v1/audio/transcriptions \
  -F file=@clip.wav -F response_format=verbose_json
```

## Models

| Checkpoint | License | Mean WER (OpenASR public) |
|---|---|---|
| `granite-speech-5.0-470m-turboctc` | Apache-2.0 | 5.00% |
| `granite-speech-5.0-470m-turboctc-nc` | CC-BY-NC-SA-4.0 | 4.85% |

Both checkpoints are functionally identical here — same service, same flags,
same GPU path. Pick purely on licensing:

- **`-nc`** (CC-BY-NC-SA-4.0) — non-commercial only, ~0.15 WER better.
- **standard** (Apache-2.0) — unrestricted, use this for anything commercial.

The example config and `k8s.yaml` default to `-nc` because this deployment is
personal. Switching is just a different `ASR_MODEL` path — no rebuild.

GGUF quants: [handy-computer/granite-speech-5.0-470m-turboctc-gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf) · […-nc-gguf](https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf)

## License

Service code: MIT. Loaded components keep their own licenses — see [LICENSE](LICENSE).