FROM ubuntu:22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

FROM base AS dev
RUN apt-get update && apt-get install -y \
    curl \
    git \
    clangd \
    pre-commit \
    && rm -rf /var/lib/apt/lists/*
RUN curl -Lo /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

FROM base AS deploy
COPY cartpole_runner /app/cartpole_runner
ENTRYPOINT ["/app/cartpole_runner"]
