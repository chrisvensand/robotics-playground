# robotics-playground

A polyglot Bazel monorepo demonstrating a commit-to-robot artifact pipeline: C++ simulation, Python ML training, pybind11 bindings, model artifact delivery, and OCI image packaging for embedded Linux targets.

## What this repo builds

Training uses MuJoCo (via pip) for high-fidelity physics simulation. The deployment binary uses lightweight custom C++ physics with no MuJoCo dependency — the same binary that would run on a constrained ARM64 robot. A trained policy weight file (`cartpole_policy.bin`) connects both worlds as a content-addressed Bazel artifact.

```
TRAINING (Python + MuJoCo)          DEPLOYMENT (C++)
────────────────────────────        ─────────────────────────────
MuJoCo high-fidelity physics        Custom CartPole physics
random_search.py                    cartpole_runner binary
     │                                      │
     └── cartpole_policy.bin ───────────────┘
              (genrule artifact)
```

## Architecture

### C++ runtime (`runtime/`)
- `runtime/sim/env.h` — abstract `sim::Env` interface (header-only). All simulations implement this.
- `runtime/cartpole/` — `CartPole : public sim::Env`, Euler integration, no external deps.
- `runtime/policy/linear_policy.{h,cc}` — loads a binary policy file, runs matrix-multiply inference.

### Python training (`tools/training/`)
- `random_search.py` — simple random search optimizer, environment-agnostic.
- `train_cartpole.py` — trains with MuJoCo, writes `cartpole_policy.bin` (obs_dim + action_dim ints, then float weights).

### pybind11 bridge (`bindings/`)
- `bindings/policy/linear_policy_py.cc` — exposes `LinearPolicy` to Python via `pybind_extension`.
- Import path: `from bindings.policy.linear_policy_py import LinearPolicy`.
- Purpose: Python validation can run the exact C++ inference code against the MuJoCo sim before shipping.

### Model artifact (`model/cartpole/`)
- `genrule` that runs `train_cartpole.py` and captures `cartpole_policy.bin` as a build output.
- The runner binary declares it as a `data` dep — Bazel wires it into runfiles automatically.

### Inference binary (`inference/cartpole_runner/`)
- `cc_binary` that loads the policy and runs the C++ CartPole sim.
- Also builds an OCI image (`cartpole_runner_image`) using `rules_oci` with `gcr.io/distroless/cc-debian12` as base.

### Platforms (`platforms/`)
- `linux_arm64` and `linux_amd64` platform definitions for cross-compilation with `toolchains_llvm`.

## Bazel setup

- **Bazel version**: pinned in `.bazelversion` (8.x — hedron uses `native.py_binary` removed in Bazel 9).
- **bzlmod**: all deps in `MODULE.bazel`, lock file in `MODULE.bazel.lock`.
- **Python**: hermetic 3.11 via `rules_python`. Never use system Python for builds.
- **C++ toolchain**: `toolchains_llvm` with LLVM 17.
- **Pip deps**: managed via `pip.parse` + `compile_pip_requirements`. Update with `bazel run //:requirements.update`.
- **Remote cache**: BuildBuddy via `.bazelrc` `remote` config. Pass API key via `.bazelrc.user` (gitignored).

## Developer environment

Open in VS Code → Reopen in Container. The devcontainer:
- Installs `bazelisk`, `clangd`, `git`, `pre-commit` (via pip for 3.x).
- Runs `bazel run @hedron_compile_commands//:refresh_all` to generate `compile_commands.json` for clangd.
- Runs `pre-commit install` so hooks run automatically on every commit.

## CI (`.github/workflows/ci.yml`)

Three jobs, all running inside the ARM64 devcontainer image published to GHCR:
1. `devcontainer` — builds and pushes `ghcr.io/<owner>/robotics-playground:devcontainer` on `ubuntu-24.04-arm`.
2. `lint` — runs `pre-commit run --all-files` inside the devcontainer.
3. `build-and-test` — runs `bazel test //...` and builds the OCI image, with BuildBuddy remote cache.

CI runs on PRs and pushes to `main` only (no duplicate runs).

## Key commands

```bash
bazel test //...                                    # run all tests
bazel build //model/cartpole:weights                # train policy
bazel run //inference/cartpole_runner:cartpole_runner  # run inference loop
bazel build //inference/cartpole_runner:cartpole_runner_image  # build OCI image
bazel run //:requirements.update                    # regenerate pip lock file
bazel run @hedron_compile_commands//:refresh_all    # regenerate compile_commands.json
```

## Common pitfalls

- `hedron_compile_commands` requires Bazel 8.x — do not upgrade `.bazelversion` to 9.x.
- pybind11 import must use the full package path: `from bindings.policy.linear_policy_py import ...`
- OCI base image is `gcr.io/distroless/cc-debian12` (not Docker Hub — rate limits cause CI timeouts).
- `.bazelrc.user` is gitignored — put your BuildBuddy API key there locally.
- `MODULE.bazel.lock` must be committed — update it after any `MODULE.bazel` change.
