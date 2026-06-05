import struct
import tempfile
import os
from bindings.policy.linear_policy_py import LinearPolicy


def write_policy(path, obs_dim, act_dim, weights):
    with open(path, "wb") as f:
        f.write(struct.pack("ii", obs_dim, act_dim))
        f.write(struct.pack(f"{len(weights)}f", *weights))


def test_zero_policy_produces_zero_action():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        path = f.name
    write_policy(path, 4, 1, [0.0, 0.0, 0.0, 0.0])
    policy = LinearPolicy.load(path)
    action = policy.act([1.0, 2.0, 3.0, 4.0])
    assert action == [0.0], f"expected [0.0], got {action}"
    os.unlink(path)


def test_known_weights():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        path = f.name
    write_policy(path, 4, 1, [1.0, 0.0, 0.0, 0.0])
    policy = LinearPolicy.load(path)
    action = policy.act([5.0, 2.0, 3.0, 4.0])
    assert action == [5.0], f"expected [5.0], got {action}"
    os.unlink(path)
