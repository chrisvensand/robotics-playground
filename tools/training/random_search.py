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
