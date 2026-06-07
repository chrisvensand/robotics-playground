"""Tests for ota_client."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.ota.ota_client import health_check, main, sha256_file, verify_signature


# ── unit tests (no system dependencies) ──────────────────────────────────────


def test_sha256_file_matches_known_hash():
    data = b"cartpole policy weights"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        path = Path(f.name)
    try:
        assert sha256_file(path) == hashlib.sha256(data).hexdigest()
    finally:
        path.unlink()


def test_verify_signature_returns_true_when_openssl_exits_zero():
    with patch("tools.ota.ota_client.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert verify_signature(Path("m.json"), Path("m.sig"), Path("pub.pem")) is True


def test_verify_signature_returns_false_when_openssl_exits_nonzero():
    with patch("tools.ota.ota_client.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        assert verify_signature(Path("m.json"), Path("m.sig"), Path("pub.pem")) is False


def test_verify_signature_passes_correct_args_to_openssl():
    with patch("tools.ota.ota_client.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        verify_signature(
            Path("manifest.json"), Path("manifest.json.sig"), Path("keys/pub.pem")
        )
    args = mock_run.call_args[0][0]
    assert args[0] == "openssl"
    assert "-verify" in args
    assert "keys/pub.pem" in args
    assert "-signature" in args
    assert "manifest.json.sig" in args
    assert "manifest.json" in args


def test_health_check_passes_when_all_files_present_and_nonempty():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        (tmpdir / "binary-v1.0.0").write_bytes(b"binary content")
        manifest = {"artifacts": [{"path": "binary-v1.0.0"}]}
        assert health_check(tmpdir, manifest) is True


def test_health_check_fails_when_file_missing():
    with tempfile.TemporaryDirectory() as d:
        manifest = {"artifacts": [{"path": "missing"}]}
        assert health_check(Path(d), manifest) is False


def test_health_check_fails_when_file_is_empty():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        (tmpdir / "empty").write_bytes(b"")
        manifest = {"artifacts": [{"path": "empty"}]}
        assert health_check(tmpdir, manifest) is False


def test_health_check_passes_with_multiple_artifacts():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        (tmpdir / "binary").write_bytes(b"bin")
        (tmpdir / "model").write_bytes(b"model")
        manifest = {"artifacts": [{"path": "binary"}, {"path": "model"}]}
        assert health_check(tmpdir, manifest) is True


# ── integration tests (require openssl on PATH) ───────────────────────────────


def _make_keypair(tmpdir: Path):
    private = tmpdir / "key.pem"
    public = tmpdir / "pub.pem"
    subprocess.run(
        ["openssl", "genrsa", "-out", str(private), "2048"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "rsa", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    return private, public


def _make_update_package(update_dir: Path, version: str, private_key: Path):
    """Create a signed update package in update_dir."""
    update_dir.mkdir(parents=True, exist_ok=True)
    data = b"fake binary"
    artifact = update_dir / f"binary-v{version}-linux-arm64"
    artifact.write_bytes(data)

    manifest = {
        "version": version,
        "created_at": "2026-01-01T00:00:00+00:00",
        "artifacts": [
            {
                "name": "binary",
                "path": artifact.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        ],
    }
    manifest_path = update_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-sign",
            str(private_key),
            "-out",
            str(update_dir / "manifest.json.sig"),
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
    )
    return manifest


def test_valid_update_installs_all_artifacts():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        private_key, public_key = _make_keypair(tmpdir)
        update_dir = tmpdir / "update"
        manifest = _make_update_package(update_dir, "1.0.0", private_key)
        install_dir = tmpdir / "install"

        with patch(
            "sys.argv",
            ["ota_client", str(update_dir), str(install_dir), str(public_key)],
        ):
            main()

        for artifact in manifest["artifacts"]:
            assert (install_dir / artifact["path"]).exists()


def test_tampered_manifest_aborts_before_install():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        private_key, public_key = _make_keypair(tmpdir)
        update_dir = tmpdir / "update"
        _make_update_package(update_dir, "1.0.0", private_key)
        install_dir = tmpdir / "install"

        # Tamper with manifest after signing — signature will no longer match
        manifest_path = update_dir / "manifest.json"
        data = json.loads(manifest_path.read_text())
        data["version"] = "tampered"
        manifest_path.write_text(json.dumps(data))

        with patch(
            "sys.argv",
            ["ota_client", str(update_dir), str(install_dir), str(public_key)],
        ):
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 1

        assert not install_dir.exists()


def test_corrupted_artifact_aborts_before_install():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        private_key, public_key = _make_keypair(tmpdir)
        update_dir = tmpdir / "update"
        manifest = _make_update_package(update_dir, "1.0.0", private_key)
        install_dir = tmpdir / "install"

        # Replace artifact with different content — SHA256 will not match manifest
        artifact_path = update_dir / manifest["artifacts"][0]["path"]
        artifact_path.write_bytes(b"corrupted content")

        with patch(
            "sys.argv",
            ["ota_client", str(update_dir), str(install_dir), str(public_key)],
        ):
            try:
                main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 1

        assert not install_dir.exists()


def test_second_update_preserves_previous_install_as_old():
    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)
        private_key, public_key = _make_keypair(tmpdir)
        install_dir = tmpdir / "install"

        update_v1 = tmpdir / "v1"
        _make_update_package(update_v1, "1.0.0", private_key)
        with patch(
            "sys.argv",
            ["ota_client", str(update_v1), str(install_dir), str(public_key)],
        ):
            main()

        update_v2 = tmpdir / "v2"
        _make_update_package(update_v2, "2.0.0", private_key)
        with patch(
            "sys.argv",
            ["ota_client", str(update_v2), str(install_dir), str(public_key)],
        ):
            main()

        # Previous install should be preserved as install.old for one cycle
        assert (tmpdir / "install.old").exists()
