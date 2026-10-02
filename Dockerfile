# Multi-stage image: the Go service, transcribe.cpp with the Vulkan backend,
# and a slim runtime. The model GGUF is mounted, never baked in.

# Service stage: compile the static Go binary.
FROM golang:1.24-alpine AS service

WORKDIR /src
COPY go.mod ./
COPY main.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/granite-asr .

# The Vulkan kernel build is slow (a few minutes) because shaderc compiles a
# large number of compute shaders, so it happens here rather than at runtime.
FROM ubuntu:24.04 AS transcribe
# Pinned to a released tag so image builds are reproducible.
ARG TRANSCRIBE_REF=v0.2.4

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates curl \
      libvulkan-dev glslc spirv-headers libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "$TRANSCRIBE_REF" \
      https://github.com/handy-computer/transcribe.cpp.git /src

RUN cmake -B build -DTRANSCRIBE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" --target transcribe-cli


# Runtime stage: the service binary, the CLI it drives, and the Vulkan/Mesa
# user-space stack. ffmpeg does container normalisation; mesa-vulkan-drivers
# provides the RADV ICD and loader that make the GPU usable.
FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg \
      libvulkan1 mesa-vulkan-drivers libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=transcribe /src/build/bin/transcribe-cli /usr/local/bin/transcribe-cli
COPY --from=service /out/granite-asr /usr/local/bin/granite-asr

# The model GGUF is mounted from a volume; it is never baked into the image.
ENV ASR_BINARY=/usr/local/bin/transcribe-cli \
    ASR_BACKEND=vulkan \
    PORT=8080

# RADV_PERFTEST=nogttspill is set in the deployment manifest rather than here so
# it stays visible in one place, but it is required on Polaris.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/granite-asr"]