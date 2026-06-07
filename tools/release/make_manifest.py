#!/usr/bin/env python3
"""Generate a release manifest with SHA256 checksums for all artifacts."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) < 3:
        print(
            f"usage: {sys.argv[0]} <version> <artifact> [<artifact>...]",
            file=sys.stderr,
        )
        sys.exit(1)

    version = sys.argv[1]
    artifact_paths = [Path(p) for p in sys.argv[2:]]

    artifacts = []
    for path in artifact_paths:
        if not path.exists():
            print(f"error: artifact not found: {path}", file=sys.stderr)
            sys.exit(1)
        artifacts.append(
            {
                "name": path.name.split("-")[0],
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
