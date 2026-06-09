import numpy as np
import mujoco

FRAME_SKIP = 10


def rollout(model, weights, rng, *, max_steps=500):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = rng.uniform(-0.1, 0.1, size=data.qpos.shape)
    data.qvel[:] = rng.uniform(-0.1, 0.1, size=data.qvel.shape)
    mujoco.mj_forward(model, data)
    total = 0.0
    for _ in range(max_steps):
        obs = [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]
        data.ctrl[0] = float(weights @ np.asarray(obs))
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
        obs = [data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]]
        done = abs(obs[0]) > 2.4 or abs(obs[2]) > 12 * 3.14159 / 180
        if done:
            break
        total += 1.0 - 0.5 * abs(obs[0]) / 2.4
    return total


def evaluate(model, weights, rng, n_rollouts=10):
    return float(np.mean([rollout(model, weights, rng) for _ in range(n_rollouts)]))


def train(model, obs_dim, action_dim, *, iterations=1000, sigma=0.5, seed=42):
    rng = np.random.default_rng(seed)
    best_w = np.zeros((action_dim, obs_dim), dtype=np.float32)
    best_return = -float("inf")
    for i in range(iterations):
        w = best_w + sigma * rng.standard_normal(best_w.shape).astype(np.float32)
        ret = evaluate(model, w, rng)
        if ret > best_return:
            best_w = w
            best_return = ret
            print(f"iter {i:4d}: return {best_return:.1f}")
    return best_w, best_return
