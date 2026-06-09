"""Render a short GIF for the README showing the policy handling perturbations."""

import os
import struct

import imageio.v3 as iio
import mujoco
import numpy as np
from tools.training.random_search import FRAME_SKIP


def _rlocation(path: str) -> str:
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if not runfiles_dir:
        f = os.path.abspath(__file__)
        marker = ".runfiles"
        idx = f.find(marker)
        if idx < 0:
            raise RuntimeError("not running inside a Bazel runfiles tree")
        runfiles_dir = f[: idx + len(marker)]
    return os.path.join(runfiles_dir, "_main", path)


def load_policy(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        obs_dim, action_dim = struct.unpack("<ii", f.read(8))
        weights = np.frombuffer(f.read(obs_dim * action_dim * 4), dtype=np.float32)
    return weights.reshape(action_dim, obs_dim)


def main() -> None:
    xml_path = _rlocation("assets/cartpole/cartpole.xml")
    policy_path = _rlocation("model/cartpole/cartpole_policy.bin")
    out_path = os.path.join(
        os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."),
        "docs/cartpole_demo.gif",
    )

    weights = load_policy(policy_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=400)

    # 15fps GIF at 2x playback speed
    fps = 15
    sim_hz = 1.0 / (FRAME_SKIP * model.opt.timestep)
    playback_speed = 2.0
    frame_every = max(1, round(sim_hz * playback_speed / fps))

    # 8 seconds of video = 120 frames at 15fps
    target_frames = 8 * fps

    rng = np.random.default_rng(7)
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.15
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.8
    cam.azimuth = 90.0
    cam.elevation = -15.0
    cam.lookat[2] = 0.4

    kick_every = int(3 * sim_hz)
    kick_magnitude = 0.7

    frames = []
    control_step = 0

    while len(frames) < target_frames:
        if control_step > 0 and control_step % kick_every == 0:
            data.qvel[1] += rng.choice([-1.0, 1.0]) * kick_magnitude

        obs = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        data.ctrl[0] = float((weights @ obs)[0])
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
        control_step += 1

        if control_step % frame_every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        if abs(data.qpos[0]) > 2.4 or abs(data.qpos[1]) > 12 * 3.14159 / 180:
            print("pole fell — policy failed")
            return

    iio.imwrite(out_path, frames, extension=".gif", fps=fps, loop=0)
    print(f"saved {len(frames)} frames → {out_path}")


if __name__ == "__main__":
    main()
