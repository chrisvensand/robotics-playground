FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    clangd \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pre-commit

RUN ARCH=$(dpkg --print-architecture) && \
    curl -Lo /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-${ARCH} \
    && chmod +x /usr/local/bin/bazel

RUN git config --system --add safe.directory '*'
