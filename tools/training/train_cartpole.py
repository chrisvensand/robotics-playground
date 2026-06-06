import struct
import sys
import mujoco
from tools.training import random_search


def main() -> None:
    out_path = sys.argv[1]
    xml_path = sys.argv[2]
    model = mujoco.MjModel.from_xml_path(xml_path)
    obs_dim, action_dim = 4, 1
    weights, ret = random_search.train(model, obs_dim, action_dim)
    print(f"training complete — final return: {ret:.1f}")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<ii", obs_dim, action_dim))
        weights.tofile(f)


if __name__ == "__main__":
    main()
