import os
import struct

import av
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
        "cartpole_eval.mp4",
    )

    weights = load_policy(policy_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    height, width = 480, 640
    renderer = mujoco.Renderer(model, height=height, width=width)

    fps = 60
    playback_speed = 3.0
    sim_hz = 1.0 / (FRAME_SKIP * model.opt.timestep)  # control steps per second
    frame_every = max(1, round(sim_hz * playback_speed / fps))

    target_minutes = 3
    target_frames = target_minutes * 60 * fps

    rng = np.random.default_rng(0)
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.15  # start with pole visibly tilted (~9 degrees)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 4.0
    cam.azimuth = 90.0
    cam.elevation = -15.0
    cam.lookat[2] = 0.3

    # Every ~5 seconds kick the pole with a sudden angular velocity impulse.
    # This is the simplest way to show the policy recovering from a disturbance:
    # no force calibration, the pole visibly swings and the policy corrects it.
    kick_every = int(5 * sim_hz)
    kick_magnitude = (
        0.7  # rad/s added to pole angular velocity — ~15 control steps to limit
    )

    control_step = 0
    frames_written = 0

    print(f"rendering {target_minutes} min at {playback_speed}× speed → {out_path}")

    container = av.open(out_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    try:
        while frames_written < target_frames:
            # Kick: add angular velocity directly to the pole
            if control_step > 0 and control_step % kick_every == 0:
                data.qvel[1] += rng.choice([-1.0, 1.0]) * kick_magnitude

            obs = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
            data.ctrl[0] = float((weights @ obs)[0])
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            control_step += 1

            if control_step % frame_every == 0:
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                frames_written += 1
                if frames_written % (fps * 10) == 0:
                    pct = 100 * frames_written / target_frames
                    sim_s = control_step / sim_hz
                    print(f"  {pct:.0f}%  sim time {sim_s:.0f}s")

            fallen = abs(data.qpos[0]) > 2.4 or abs(data.qpos[1]) > 12 * 3.14159 / 180
            if fallen:
                sim_s = control_step / sim_hz
                print(f"pole fell at {sim_s:.1f}s — policy failed")
                break

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    print(f"done — {frames_written} frames → {out_path}")


if __name__ == "__main__":
    main()
