import struct
import sys


def main():
    obs_dim = 4
    act_dim = 1
    weights = [0.0] * (obs_dim * act_dim)
    data = struct.pack(f"{len(weights)}f", *weights)
    path = sys.argv[1] if len(sys.argv) > 1 else "zero_policy.bin"
    with open(path, "wb") as f:
        f.write(data)
    print(f"wrote {len(weights)} floats to {path}")


if __name__ == "__main__":
    main()
