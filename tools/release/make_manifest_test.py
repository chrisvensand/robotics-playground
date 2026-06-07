"""Tests for make_manifest."""

import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tools.release.make_manifest import main, sha256_file


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_main(*args) -> dict:
    """Run main() with the given args and return parsed JSON output."""
    with patch("sys.argv", ["make_manifest", *args]):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            main()
    return json.loads(mock_out.getvalue())


def test_sha256_file_matches_known_hash():
    data = b"cartpole policy weights"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        path = Path(f.name)
    try:
        assert sha256_file(path) == _sha256(data)
    finally:
        path.unlink()


def test_sha256_file_empty():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = Path(f.name)
    try:
        assert sha256_file(path) == _sha256(b"")
    finally:
        path.unlink()


def test_manifest_contains_version():
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / "binary-v2.3.4-linux-arm64"
        artifact.write_bytes(b"data")
        manifest = _run_main("2.3.4", str(artifact))
    assert manifest["version"] == "2.3.4"


def test_manifest_contains_created_at():
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / "binary-v1.0.0-linux-arm64"
        artifact.write_bytes(b"data")
        manifest = _run_main("1.0.0", str(artifact))
    assert "created_at" in manifest
    assert manifest["created_at"]


def test_artifact_name_uses_prefix_before_first_dash():
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / "cartpole_runner-v1.0.0-linux-arm64"
        artifact.write_bytes(b"data")
        manifest = _run_main("1.0.0", str(artifact))
    assert manifest["artifacts"][0]["name"] == "cartpole_runner"


def test_artifact_path_is_filename():
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / "cartpole_runner-v1.0.0-linux-arm64"
        artifact.write_bytes(b"data")
        manifest = _run_main("1.0.0", str(artifact))
    assert manifest["artifacts"][0]["path"] == "cartpole_runner-v1.0.0-linux-arm64"


def test_artifact_sha256_matches_file_contents():
    data = b"known binary content"
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / "runner-v1.0.0-linux-arm64"
        artifact.write_bytes(data)
        manifest = _run_main("1.0.0", str(artifact))
    assert manifest["artifacts"][0]["sha256"] == _sha256(data)


def test_artifact_size_bytes_matches_file_size():
    data = b"exactly this many bytes"
    with tempfile.TemporaryDirectory() as d:
        artifact = Path(d) / "runner-v1.0.0-linux-arm64"
        artifact.write_bytes(data)
        manifest = _run_main("1.0.0", str(artifact))
    assert manifest["artifacts"][0]["size_bytes"] == len(data)


def test_multiple_artifacts_all_included():
    with tempfile.TemporaryDirectory() as d:
        a1 = Path(d) / "cartpole_runner-v1.0.0-linux-arm64"
        a2 = Path(d) / "cartpole_policy-v1.0.0.bin"
        a1.write_bytes(b"runner")
        a2.write_bytes(b"policy")
        manifest = _run_main("1.0.0", str(a1), str(a2))
    assert len(manifest["artifacts"]) == 2
    names = {a["name"] for a in manifest["artifacts"]}
    assert names == {"cartpole_runner", "cartpole_policy"}


def test_missing_artifact_exits_nonzero():
    with patch("sys.argv", ["make_manifest", "1.0.0", "/nonexistent/artifact"]):
        try:
            main()
            assert False, "expected SystemExit"
        except SystemExit as e:
            assert e.code == 1
