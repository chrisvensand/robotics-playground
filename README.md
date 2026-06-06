# robotics-playground

A polyglot Bazel monorepo demonstrating a commit-to-robot artifact pipeline: C++ simulation, Python ML training, pybind11 bindings, model artifact delivery, and OCI image packaging for embedded Linux targets.

## Architecture

```
TRAINING (Python + MuJoCo)          DEPLOYMENT (C++)
────────────────────────────        ─────────────────────────────
MuJoCo high-fidelity physics        Custom CartPole physics
random_search.py                    cartpole_runner binary
     │                                      │
     └── cartpole_policy.bin ───────────────┘
              (genrule artifact)
```

Training uses MuJoCo (via pip) for high-fidelity physics. The deployment binary uses lightweight custom C++ physics — no MuJoCo dependency on the robot. The policy weights connect both worlds as a content-addressed Bazel artifact.

## Getting started

Open the repo in VS Code and click **Reopen in Container** when prompted. The devcontainer builds the development environment, installs clangd, generates `compile_commands.json` for IDE support, and installs pre-commit hooks — no manual setup required.

## Build and run

```bash
# Run all tests (C++ + Python)
bazel test //...

# Train the policy and produce cartpole_policy.bin
bazel build //model/cartpole:weights

# Run the cartpole inference loop
bazel run //inference/cartpole_runner:cartpole_runner

# Build and run the OCI image
bazel build //inference/cartpole_runner:cartpole_runner_tarball
docker load < bazel-bin/inference/cartpole_runner/cartpole_runner_tarball/tarball.tar
docker run --rm cartpole_runner:latest
```

## CI

GitHub Actions runs on every push and pull request:

1. **lint** — pre-commit (clang-format, ruff, file hygiene)
2. **build-and-test** — `bazel test //...`, cross-compile to `linux/arm64`, OCI image build

## What each component demonstrates

| Component | Concept |
|-----------|---------|
| `MODULE.bazel` + bzlmod | Modern Bazel dependency management |
| `sim::Env` interface | Extensible multi-sim framework |
| `rules_python` hermetic toolchain | Single Python version across all developers and CI |
| `pip.parse` + `compile_pip_requirements` | Hash-pinned pip dependencies managed by Bazel |
| `pybind11_bazel` | C++ ↔ Python bridge — validate policy with C++ inference code during training |
| `genrule` policy artifact | Trained model as a content-addressed build artifact |
| `rules_oci` + `rules_pkg` | OCI image built entirely in Bazel — no Docker daemon required |
| `toolchains_llvm` + `--platforms` | Cross-compilation from Linux AMD64 → Linux ARM64 |
| `setup-bazel` action | Cached incremental CI builds via GitHub Actions |
