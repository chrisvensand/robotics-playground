FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    clangd \
    jq \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pre-commit

RUN ARCH=$(dpkg --print-architecture) && \
    curl -Lo /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-${ARCH} \
    && chmod +x /usr/local/bin/bazel

RUN ARCH=$(dpkg --print-architecture) && \
    curl -Lo /usr/local/bin/buildifier \
    https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-linux-${ARCH} \
    && chmod +x /usr/local/bin/buildifier

RUN ARCH=$(dpkg --print-architecture) && \
    curl -Lo /usr/local/bin/cosign \
    https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-${ARCH} \
    && chmod +x /usr/local/bin/cosign

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory '*'
