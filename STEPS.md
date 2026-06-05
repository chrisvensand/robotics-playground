# Bazel + Robotics Playground — Learning Plan

Goal: build a Bazel project from scratch that integrates C++ with Python ML models, mirroring the commit-to-robot pipeline at Sunday Robotics. The project is a **multi-simulation framework**: cartpole first, robot arm later.

Each stage maps to a JD bullet so I can articulate it in the interview.

## Roadmap

| Stage | What | JD bullet |
|-------|------|-----------|
| 1 | C++ Bazel fundamentals; `Env` interface; custom CartPole physics; `cc_test` | Build system |
| 2 | Python in the same repo; `mujoco` pip package for high-fidelity training | Polyglot, dev env |
| 3 | pybind11 — expose C++ `LinearPolicy` to Python for validation | ML alongside C++ runtime |
| 4 | Train policy with MuJoCo; export as Bazel build artifact; load in C++ runner | Model delivery |
| 5 | OCI image, cross-compile to arm64, CI, signing | Artifact pipeline, OTA, reproducible builds, signing |

Working rule: hand-type all code and run all commands myself. Claude is in tutor mode — it walks me through one step at a time and explains *why*, but I do the typing.

---

## Architecture

Two separate worlds connected by a build artifact:

```
TRAINING SIDE (Python)                  DEPLOYMENT SIDE (C++)
──────────────────────────────          ──────────────────────────────
MuJoCo sim (pip package)                Custom CartPole physics
  └─ high-fidelity physics                └─ lightweight, no external deps
  └─ cartpole.xml model file               └─ runs on constrained hardware

random_search.py trains policy          cartpole_runner binary
  └─ writes cartpole_policy.bin    →→→    └─ loads cartpole_policy.bin
                                            └─ runs inference loop
pybind11 bridge ──────────────────────────────────────────────────────
  └─ exposes C++ LinearPolicy to Python
  └─ validation: train with MuJoCo, evaluate with C++ policy code
  └─ closes the training-deployment gap without coupling the sims
```

**Why custom C++ physics instead of MuJoCo in C++?**
MuJoCo is a training tool. The deployed binary loads a trained policy and runs it against real robot sensor data — it doesn't simulate anything. Shipping MuJoCo's full library to constrained hardware is the wrong tradeoff. This is the architecture real manipulation teams use.

**Where pybind11 fits:**
The pybind11 bridge exposes `LinearPolicy.act()` to Python. Validation runs the trained policy through the exact same C++ evaluation code that ships on the robot — but against the MuJoCo sim. If validation passes, you know the policy works with the actual inference code.

---

## Project structure

```
/
├── MODULE.bazel               # bzlmod entry; declares all deps
├── .bazelversion
├── .gitignore
├── requirements.txt           # Python pip deps (mujoco, numpy, pytest)
├── assets/                    # simulation model files for Python training
│   └── cartpole/
│       ├── cartpole.xml       # MuJoCo model — data dep for Python training
│       └── BUILD.bazel
├── platforms/                 # Bazel platform definitions (Stage 5)
├── runtime/                   # C++ framework (no MuJoCo dependency)
│   ├── sim/                   # Env interface (header-only cc_library)
│   ├── policy/                # LinearPolicy loader
│   └── cartpole/              # CartPole : public sim::Env, custom physics
├── bindings/                  # pybind11 bridges
│   └── policy/                # exposes C++ LinearPolicy to Python
├── tools/                     # Python training + validation scripts
│   └── training/
│       ├── train_cartpole.py  # trains with MuJoCo, writes policy.bin
│       └── validate_cartpole.py  # validates with C++ policy + MuJoCo sim
├── model/                     # genrule: training → policy artifact
│   └── cartpole/
├── inference/                 # deployable C++ binaries (no MuJoCo)
│   └── cartpole_runner/
└── .github/workflows/
```

### How to add a new simulation

1. Add XML to `assets/<sim>/` for MuJoCo training.
2. `runtime/<sim>/` — custom C++ physics implementing `sim::Env` for deployment.
3. `tools/training/train_<sim>.py` — training with MuJoCo pip + `random_search.py`.
4. `tools/training/validate_<sim>.py` — validate using C++ policy via pybind11.
5. `model/<sim>/BUILD.bazel` — genrule capturing `<sim>_policy.bin`.
6. `inference/<sim>_runner/` — `cc_binary` loading policy + running custom C++ sim.

---

## Stage 1 — C++ Bazel fundamentals

### 1.A Setup + first binary  ✓ DONE

- `.bazelversion` = 9.1.1
- `MODULE.bazel` with `rules_cc = 0.2.19`
- `inference/cartpole_runner/main.cc` + `BUILD.bazel`
- `bazel build`, `bazel run`, `bazel query` all working

---

### 1.B Define the `Env` interface

**Step 1.B.1 — Create `runtime/sim/`**

```bash
mkdir -p runtime/sim
```

Hand-write `runtime/sim/env.h`:

```cpp
#pragma once
#include <vector>

namespace sim {

struct Step {
    std::vector<float> observation;
    float reward;
    bool done;
};

class Env {
public:
    virtual ~Env() = default;
    virtual std::vector<float> reset() = 0;
    virtual Step step(const std::vector<float>& action) = 0;
    virtual int observation_dim() const = 0;
    virtual int action_dim() const = 0;
};

}  // namespace sim
```

Hand-write `runtime/sim/BUILD.bazel`:

```python
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "env",
    hdrs = ["env.h"],
    visibility = ["//visibility:public"],
)
```

Header-only library — `hdrs` but no `srcs`. Nothing to compile; just an interface.

---

### 1.C Implement CartPole with custom C++ physics

**Step 1.C.1 — Create `runtime/cartpole/`**

```bash
mkdir -p runtime/cartpole
```

Hand-write `runtime/cartpole/cartpole.h`:

```cpp
#pragma once
#include "runtime/sim/env.h"

namespace cartpole {

class CartPole : public sim::Env {
public:
    CartPole();
    std::vector<float> reset() override;
    sim::Step step(const std::vector<float>& action) override;
    int observation_dim() const override { return 4; }
    int action_dim() const override { return 1; }

private:
    float x_, x_dot_, theta_, theta_dot_;
    int step_count_;
};

}  // namespace cartpole
```

Hand-write `runtime/cartpole/cartpole.cc`:

```cpp
#include "runtime/cartpole/cartpole.h"
#include <cmath>

namespace cartpole {

namespace {
constexpr float kGravity = 9.8f;
constexpr float kCartMass = 1.0f;
constexpr float kPoleMass = 0.1f;
constexpr float kPoleLength = 0.5f;
constexpr float kForceMag = 10.0f;
constexpr float kTau = 0.02f;
constexpr float kThetaThreshold = 12.0f * 3.14159265f / 180.0f;
constexpr float kXThreshold = 2.4f;
constexpr int kMaxSteps = 500;
}

CartPole::CartPole() { reset(); }

std::vector<float> CartPole::reset() {
    x_ = x_dot_ = theta_ = theta_dot_ = 0.0f;
    step_count_ = 0;
    return {x_, x_dot_, theta_, theta_dot_};
}

sim::Step CartPole::step(const std::vector<float>& action) {
    float force = (action[0] > 0.0f) ? kForceMag : -kForceMag;
    float costheta = std::cos(theta_);
    float sintheta = std::sin(theta_);
    float total_mass = kCartMass + kPoleMass;
    float pole_ml = kPoleMass * kPoleLength;
    float temp = (force + pole_ml * theta_dot_ * theta_dot_ * sintheta) / total_mass;
    float thetaacc = (kGravity * sintheta - costheta * temp) /
        (kPoleLength * (4.0f / 3.0f - kPoleMass * costheta * costheta / total_mass));
    float xacc = temp - pole_ml * thetaacc * costheta / total_mass;
    x_        += kTau * x_dot_;
    x_dot_    += kTau * xacc;
    theta_    += kTau * theta_dot_;
    theta_dot_+= kTau * thetaacc;
    step_count_++;
    bool done = std::abs(x_) > kXThreshold
             || std::abs(theta_) > kThetaThreshold
             || step_count_ >= kMaxSteps;
    return {{x_, x_dot_, theta_, theta_dot_}, 1.0f, done};
}

}  // namespace cartpole
```

Hand-write `runtime/cartpole/BUILD.bazel`:

```python
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "cartpole",
    srcs = ["cartpole.cc"],
    hdrs = ["cartpole.h"],
    deps = ["//runtime/sim:env"],
    visibility = ["//visibility:public"],
)
```

**Step 1.C.2 — Update the runner to use CartPole**

Replace `inference/cartpole_runner/main.cc`:

```cpp
#include "runtime/cartpole/cartpole.h"
#include <cstdio>

int main() {
    cartpole::CartPole env;
    auto obs = env.reset();
    std::printf("initial: x=%.3f x_dot=%.3f theta=%.3f theta_dot=%.3f\n",
                obs[0], obs[1], obs[2], obs[3]);
    auto step = env.step({1.0f});
    std::printf("after step: x=%.3f theta=%.3f done=%d\n",
                step.observation[0], step.observation[2], step.done);
    return 0;
}
```

Update `inference/cartpole_runner/BUILD.bazel`:

```python
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "cartpole_runner",
    srcs = ["main.cc"],
    deps = ["//runtime/cartpole:cartpole"],
)
```

```bash
bazel run //inference/cartpole_runner:cartpole_runner
```

---

### 1.D Add a `cc_test` for CartPole

**Step 1.D.1 — Add googletest to `MODULE.bazel`**

```python
bazel_dep(name = "googletest", version = "1.15.2")
```

**Step 1.D.2 — Hand-write `runtime/cartpole/cartpole_test.cc`**

```cpp
#include "runtime/cartpole/cartpole.h"
#include "gtest/gtest.h"

TEST(CartPoleTest, ResetReturnsZeros) {
    cartpole::CartPole env;
    auto obs = env.reset();
    ASSERT_EQ(obs.size(), 4);
    for (float v : obs) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(CartPoleTest, StepChangesState) {
    cartpole::CartPole env;
    env.reset();
    auto step = env.step({1.0f});
    EXPECT_GT(step.observation[1], 0.0f);
    EXPECT_FLOAT_EQ(step.reward, 1.0f);
    EXPECT_FALSE(step.done);
}

TEST(CartPoleTest, EpisodeEnds) {
    cartpole::CartPole env;
    env.reset();
    sim::Step step;
    for (int i = 0; i < 200; ++i) {
        step = env.step({1.0f});
        if (step.done) break;
    }
    EXPECT_TRUE(step.done);
}
```

**Step 1.D.3 — Add `cc_test` to `runtime/cartpole/BUILD.bazel`**

```python
cc_test(
    name = "cartpole_test",
    srcs = ["cartpole_test.cc"],
    deps = [
        ":cartpole",
        "@googletest//:gtest_main",
    ],
)
```

```bash
bazel test //runtime/cartpole:cartpole_test
bazel test //...
```

---

## Stage 2 — Python + MuJoCo for training

### 2.A `rules_python` + hermetic interpreter

**Step 2.A.1 — Add to `MODULE.bazel`**

```python
bazel_dep(name = "rules_python", version = "0.36.0")

python = use_extension("@rules_python//python/extensions:python.bzl", "python")
python.toolchain(python_version = "3.11")
```

**Step 2.A.2 — Hand-write `requirements.txt`**

```
mujoco>=3.0.0
numpy>=2.0.0
pytest>=8.0.0
```

**Step 2.A.3 — Wire pip into `MODULE.bazel`**

```python
pip = use_extension("@rules_python//python/extensions:pip.bzl", "pip")
pip.parse(
    hub_name = "pip",
    python_version = "3.11",
    requirements_lock = "//:requirements.txt",
)
use_repo(pip, "pip")
```

### 2.B Add the cartpole XML model

```bash
mkdir -p assets/cartpole
```

Hand-write `assets/cartpole/cartpole.xml`:

```xml
<mujoco model="cartpole">
  <compiler autolimits="true"/>
  <default>
    <joint damping="0.05"/>
  </default>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.4 2.4"/>
      <geom type="box" size="0.1 0.05 0.05"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide" joint="slider" gear="100"/>
  </actuator>
</mujoco>
```

Hand-write `assets/cartpole/BUILD.bazel`:

```python
filegroup(
    name = "cartpole_xml",
    srcs = ["cartpole.xml"],
    visibility = ["//visibility:public"],
)
```

### 2.C A `py_binary` that does a random rollout with MuJoCo

```bash
mkdir -p tools/training
```

Hand-write `tools/training/random_rollout.py`:

```python
"""Random policy rollout using MuJoCo — baseline before training."""
import random
import mujoco

def main() -> None:
    model = mujoco.MjModel.from_xml_path("assets/cartpole/cartpole.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    steps = 0
    while True:
        data.ctrl[0] = random.choice([-1.0, 1.0])
        mujoco.mj_step(model, data)
        steps += 1
        x, theta = data.qpos[0], data.qpos[1]
        if abs(x) > 2.4 or abs(theta) > 12 * 3.14159 / 180 or steps >= 500:
            break
    print(f"random policy: survived {steps} steps")

if __name__ == "__main__":
    main()
```

Hand-write `tools/training/BUILD.bazel`:

```python
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

py_binary(
    name = "random_rollout",
    srcs = ["random_rollout.py"],
    main = "random_rollout.py",
    data = ["//assets/cartpole:cartpole_xml"],
    deps = ["@pip//mujoco"],
)
```

```bash
bazel run //tools/training:random_rollout
```

### 2.D `py_test`

Hand-write `tools/training/mujoco_test.py`:

```python
import mujoco

def test_model_loads():
    model = mujoco.MjModel.from_xml_path("assets/cartpole/cartpole.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    assert data.qpos[0] == 0.0

def test_step_moves_cart():
    model = mujoco.MjModel.from_xml_path("assets/cartpole/cartpole.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.ctrl[0] = 1.0
    mujoco.mj_step(model, data)
    assert data.qvel[0] > 0.0
```

Add to `tools/training/BUILD.bazel`:

```python
py_test(
    name = "mujoco_test",
    srcs = ["mujoco_test.py"],
    data = ["//assets/cartpole:cartpole_xml"],
    deps = ["@pip//mujoco", "@pip//pytest"],
)
```

```bash
bazel test //tools/training:mujoco_test
bazel test //...
```

One `bazel test //...` now runs C++ and Python tests. Polyglot win.

---

## Stage 3 — pybind11: expose C++ `LinearPolicy` to Python

Goal: Python validation code calls the same C++ `LinearPolicy.act()` that ships on the robot. This closes the training-deployment gap — you know the trained policy works with the actual inference code before it ships.

### 3.A Create the `LinearPolicy` C++ library

**Step 3.A.1 — Create `runtime/policy/`**

```bash
mkdir -p runtime/policy
```

Hand-write `runtime/policy/linear_policy.h`:

```cpp
#pragma once
#include <string>
#include <vector>

namespace policy {

class LinearPolicy {
public:
    static LinearPolicy load(const std::string& path);
    std::vector<float> act(const std::vector<float>& observation) const;
    int observation_dim() const { return obs_dim_; }
    int action_dim() const { return action_dim_; }

private:
    int obs_dim_ = 0;
    int action_dim_ = 0;
    std::vector<float> weights_;
};

}  // namespace policy
```

Hand-write `runtime/policy/linear_policy.cc`:

```cpp
#include "runtime/policy/linear_policy.h"
#include <fstream>
#include <stdexcept>

namespace policy {

LinearPolicy LinearPolicy::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    LinearPolicy p;
    f.read(reinterpret_cast<char*>(&p.obs_dim_), sizeof(int));
    f.read(reinterpret_cast<char*>(&p.action_dim_), sizeof(int));
    p.weights_.resize(p.obs_dim_ * p.action_dim_);
    f.read(reinterpret_cast<char*>(p.weights_.data()),
           p.weights_.size() * sizeof(float));
    return p;
}

std::vector<float> LinearPolicy::act(const std::vector<float>& obs) const {
    std::vector<float> out(action_dim_, 0.0f);
    for (int a = 0; a < action_dim_; ++a)
        for (int o = 0; o < obs_dim_; ++o)
            out[a] += weights_[a * obs_dim_ + o] * obs[o];
    return out;
}

}  // namespace policy
```

Hand-write `runtime/policy/BUILD.bazel`:

```python
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "linear_policy",
    srcs = ["linear_policy.cc"],
    hdrs = ["linear_policy.h"],
    visibility = ["//visibility:public"],
)
```

### 3.B Wrap LinearPolicy in a pybind11 module

**Step 3.B.1 — Add `pybind11_bazel` to `MODULE.bazel`**

```python
bazel_dep(name = "pybind11_bazel", version = "2.13.6")
```

**Step 3.B.2 — Create `bindings/policy/`**

```bash
mkdir -p bindings/policy
```

Hand-write `bindings/policy/linear_policy_py.cc`:

```cpp
#include "runtime/policy/linear_policy.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(linear_policy_py, m) {
    py::class_<policy::LinearPolicy>(m, "LinearPolicy")
        .def_static("load", &policy::LinearPolicy::load)
        .def("act", &policy::LinearPolicy::act)
        .def("observation_dim", &policy::LinearPolicy::observation_dim)
        .def("action_dim", &policy::LinearPolicy::action_dim);
}
```

Hand-write `bindings/policy/BUILD.bazel`:

```python
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "linear_policy_py",
    srcs = ["linear_policy_py.cc"],
    deps = ["//runtime/policy:linear_policy"],
    visibility = ["//visibility:public"],
)
```

### 3.C Validate using C++ policy + MuJoCo sim

Hand-write `tools/training/validate_cartpole.py`:

```python
"""Validate a trained policy using C++ LinearPolicy evaluation + MuJoCo sim."""
import sys
import mujoco
from bindings.policy import linear_policy_py

def main() -> None:
    policy_path = sys.argv[1]
    policy = linear_policy_py.LinearPolicy.load(policy_path)

    model = mujoco.MjModel.from_xml_path("assets/cartpole/cartpole.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    steps = 0
    while True:
        obs = [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]
        action = policy.act(obs)   # C++ inference code
        data.ctrl[0] = action[0]
        mujoco.mj_step(model, data)
        steps += 1
        done = (abs(data.qpos[0]) > 2.4
                or abs(data.qpos[1]) > 12 * 3.14159 / 180
                or steps >= 500)
        if done:
            break

    print(f"validation: {steps} steps (500 = perfect)")

if __name__ == "__main__":
    main()
```

Add to `tools/training/BUILD.bazel`:

```python
py_binary(
    name = "validate_cartpole",
    srcs = ["validate_cartpole.py"],
    main = "validate_cartpole.py",
    data = ["//assets/cartpole:cartpole_xml"],
    deps = [
        "//bindings/policy:linear_policy_py",
        "@pip//mujoco",
    ],
)
```

---

## Stage 4 — Train a policy and ship it as a build artifact

### 4.A Write the training script

**Step 4.A.1 — Shared random search**

Hand-write `tools/training/random_search.py`:

```python
"""Random-search optimizer. Works with any MuJoCo environment."""
import numpy as np
import mujoco

def rollout(model, weights, *, max_steps=500):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    obs = [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]
    total = 0.0
    for _ in range(max_steps):
        action = (weights @ np.asarray(obs)).tolist()
        data.ctrl[0] = action[0]
        mujoco.mj_step(model, data)
        obs = [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]
        total += 1.0
        done = abs(obs[0]) > 2.4 or abs(obs[2]) > 12 * 3.14159 / 180
        if done:
            break
    return total

def train(model, obs_dim, action_dim, *, iterations=300, sigma=0.5, seed=42):
    rng = np.random.default_rng(seed)
    best_w = np.zeros((action_dim, obs_dim), dtype=np.float32)
    best_return = -float("inf")
    for i in range(iterations):
        w = best_w + sigma * rng.standard_normal(best_w.shape).astype(np.float32)
        ret = rollout(model, w)
        if ret > best_return:
            best_w = w
            best_return = ret
            print(f"iter {i:4d}: return {best_return:.1f}")
    return best_w, best_return
```

**Step 4.A.2 — Cartpole training entry point**

Hand-write `tools/training/train_cartpole.py`:

```python
"""Train a linear policy for cartpole and write weights to disk."""
import struct
import sys
import numpy as np
import mujoco
from tools.training import random_search

def main() -> None:
    out_path = sys.argv[1]
    model = mujoco.MjModel.from_xml_path("assets/cartpole/cartpole.xml")
    obs_dim, action_dim = 4, 1
    weights, ret = random_search.train(model, obs_dim, action_dim)
    print(f"training complete — final return: {ret:.1f}")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<ii", obs_dim, action_dim))
        weights.tofile(f)

if __name__ == "__main__":
    main()
```

Add to `tools/training/BUILD.bazel`:

```python
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "random_search",
    srcs = ["random_search.py"],
    deps = ["@pip//numpy", "@pip//mujoco"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "train_cartpole",
    srcs = ["train_cartpole.py"],
    main = "train_cartpole.py",
    data = ["//assets/cartpole:cartpole_xml"],
    deps = [
        ":random_search",
        "@pip//mujoco",
        "@pip//numpy",
    ],
    visibility = ["//visibility:public"],
)
```

### 4.B Wrap training in a `genrule`

```bash
mkdir -p model/cartpole
```

Hand-write `model/cartpole/BUILD.bazel`:

```python
genrule(
    name = "weights",
    outs = ["cartpole_policy.bin"],
    cmd = "$(execpath //tools/training:train_cartpole) $@",
    tools = ["//tools/training:train_cartpole"],
    visibility = ["//visibility:public"],
)
```

```bash
bazel build //model/cartpole:weights
```

The policy is now a content-addressed Bazel artifact — same inputs, same output, every time.

### 4.C Wire the policy into the C++ runner

Replace `inference/cartpole_runner/main.cc`:

```cpp
#include "runtime/cartpole/cartpole.h"
#include "runtime/policy/linear_policy.h"
#include <cstdio>

int main() {
    cartpole::CartPole env;
    auto pol = policy::LinearPolicy::load("model/cartpole/cartpole_policy.bin");
    auto obs = env.reset();
    int steps = 0;
    float total = 0.0f;
    while (true) {
        auto action = pol.act(obs);
        auto step = env.step(action);
        obs = step.observation;
        total += step.reward;
        steps++;
        if (step.done) break;
    }
    std::printf("cartpole_runner: %d steps, reward %.1f\n", steps, total);
    return 0;
}
```

Update `inference/cartpole_runner/BUILD.bazel`:

```python
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "cartpole_runner",
    srcs = ["main.cc"],
    data = ["//model/cartpole:weights"],
    deps = [
        "//runtime/cartpole:cartpole",
        "//runtime/policy:linear_policy",
    ],
)
```

```bash
bazel run //inference/cartpole_runner:cartpole_runner
```

Trained policy should survive hundreds of steps. Pipeline complete: MuJoCo training → policy.bin → C++ runner.

---

## Stage 5 — Robot-shaped delivery

### 5.A OCI image with `rules_oci`

**Step 5.A.1 — Add to `MODULE.bazel`**

```python
bazel_dep(name = "rules_oci", version = "2.0.0")
bazel_dep(name = "rules_pkg", version = "1.0.1")
```

**Step 5.A.2 — Pull a base image**

```python
oci = use_extension("@rules_oci//oci:extensions.bzl", "oci")
oci.pull(
    name = "distroless_cc",
    image = "gcr.io/distroless/cc-debian12",
    platforms = ["linux/amd64", "linux/arm64"],
)
use_repo(oci, "distroless_cc", "distroless_cc_linux_amd64", "distroless_cc_linux_arm64")
```

**Step 5.A.3 — Package the runner into an OCI image**

Add to `inference/cartpole_runner/BUILD.bazel`:

```python
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

pkg_tar(
    name = "runner_layer",
    srcs = [":cartpole_runner"],
    package_dir = "/app",
)

oci_image(
    name = "cartpole_runner_image",
    base = "@distroless_cc",
    entrypoint = ["/app/cartpole_runner"],
    tars = [":runner_layer"],
)
```

```bash
bazel build //inference/cartpole_runner:cartpole_runner_image
```

### 5.B Cross-compile to linux/arm64

**Step 5.B.1 — Add `toolchains_llvm` to `MODULE.bazel`**

```python
bazel_dep(name = "toolchains_llvm", version = "1.2.0")
llvm = use_extension("@toolchains_llvm//toolchain/extensions:llvm.bzl", "llvm")
llvm.toolchain(llvm_version = "17.0.6")
use_repo(llvm, "llvm_toolchain")
register_toolchains("@llvm_toolchain//:all")
```

**Step 5.B.2 — Define the target platform**

```bash
mkdir platforms
```

Hand-write `platforms/BUILD.bazel`:

```python
platform(
    name = "linux_arm64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)
```

**Step 5.B.3 — Build for linux/arm64**

```bash
bazel build //inference/cartpole_runner:cartpole_runner --platforms=//platforms:linux_arm64
```

The C++ runner has no external C++ deps beyond `rules_cc` — no MuJoCo, no nothing. Cross-compiling a lean binary is straightforward.

### 5.C CI with GitHub Actions

```bash
mkdir -p .github/workflows
```

Hand-write `.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bazel-contrib/setup-bazel@0.9.0
        with:
          bazelisk-cache: true
          disk-cache: ${{ github.workflow }}
          repository-cache: true
      - run: bazel test //...
      - run: bazel build //inference/cartpole_runner:cartpole_runner_image
```

### 5.D Image signing with cosign (bonus)

```yaml
      - uses: sigstore/cosign-installer@v3
      - run: cosign sign --yes ghcr.io/<user>/cartpole_runner:${{ github.sha }}
```

---

## What to talk about in the interview

- **Hermetic builds:** bazelisk + `.bazelversion` + bzlmod + hermetic Python + LLVM toolchain.
- **Polyglot graph:** `bazel test //...` runs C++ and Python tests in one command.
- **Training vs. deployment split:** MuJoCo (pip) for high-fidelity training; custom C++ physics for the lean deployment binary. This is the right tradeoff for constrained robot hardware — you don't ship a full physics engine to the robot.
- **pybind11 bridge for validation:** Python validation calls C++ `LinearPolicy.act()` — the exact same inference code that ships to the robot. If it passes validation, you know the policy works end-to-end before it ships.
- **Model as artifact:** `genrule`-produced `cartpole_policy.bin` is content-addressed and a `data` dep of the runner. Versioning falls out of the build graph.
- **Clean cross-compile:** because the C++ runner has no external C++ deps, `--platforms=//platforms:linux_arm64` just works.
- **OCI without Docker:** `rules_oci` builds reproducible images from the build graph.
- **Signed artifacts:** `cosign` ties image digest to supply-chain verification.
- **OTA story (talking point):** the OCI image + signed manifest is the natural unit for staged rollout, canarying, and rollback.
- **Multi-sim framework:** `sim::Env` interface means adding a robot arm follows a fixed recipe.

---

## Bazel glossary (interview reference)

- **Module:** the unit of dependency in bzlmod. Defined by `MODULE.bazel`.
- **Package:** a directory containing a `BUILD.bazel` file.
- **Target:** a thing the build graph can produce.
- **Label:** `//runtime/cartpole:cartpole` = package `runtime/cartpole`, target `cartpole`.
- **Rule:** a function that defines how to produce a target (`cc_binary`, `genrule`).
- **Ruleset:** a module that exposes rules (`rules_cc`, `rules_python`, `rules_oci`).
- **Toolchain:** Bazel's abstraction for compilers/interpreters. Selected per platform.
- **Action graph:** the DAG of actions Bazel computes from BUILD files. Caching is keyed by inputs.
- **Hermetic build:** same inputs always produce the same outputs, regardless of host environment.
- **Runfiles:** the directory of files a binary needs at runtime. Populated by `data = [...]`.
- **`select({})`:** chooses different values based on target platform or build config.
- **`bazel query`:** `deps(X)`, `rdeps(X, Y)`, `kind("cc_test", //...)`.
- **Visibility:** which packages may depend on a target.
- **Genrule:** runs a shell command at build time and captures outputs into the build graph.
- **`filegroup`:** exposes non-code files as Bazel targets for `data` deps.
- **BCR:** `registry.bazel.build` — the index of public Bazel modules.
- **`http_archive`:** fetches a tarball and makes it available as an external dep.
- **`cc_import`:** declares a pre-built C/C++ library as a Bazel target.
