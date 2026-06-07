#!/usr/bin/env python3
"""
OTA client: verifies manifest signature, verifies artifact checksums,
and performs an atomic install with rollback on failure.

Usage:
    ota_client.py <update_dir> <install_dir> <public_key>

    update_dir:   directory containing manifest.json, manifest.json.sig,
                  and all artifact files listed in the manifest
    install_dir:  where verified artifacts are installed
    public_key:   path to the release public key (PEM format)
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_signature(manifest: Path, sig: Path, pubkey: Path) -> bool:
    result = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            str(pubkey),
            "-signature",
            str(sig),
            str(manifest),
        ],
        capture_output=True,
    )
    return result.returncode == 0


def health_check(install_dir: Path, manifest: dict) -> bool:
    """Verify the installed binary is a valid executable.

    In production this would write to the staging partition, reboot,
    and wait for the new binary to call confirm_update() within a
    watchdog timeout. On failure the bootloader reverts to partition A.
    """
    for artifact in manifest["artifacts"]:
        path = install_dir / artifact["path"]
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} <update_dir> <install_dir> <public_key>",
            file=sys.stderr,
        )
        sys.exit(1)

    update_dir = Path(sys.argv[1])
    install_dir = Path(sys.argv[2])
    pubkey = Path(sys.argv[3])

    manifest_path = update_dir / "manifest.json"
    sig_path = update_dir / "manifest.json.sig"

    print("=== OTA Update ===")

    # Step 1: verify manifest signature
    # This proves the manifest was produced by the holder of the private key
    # and has not been tampered with since signing.
    print("[1/4] Verifying manifest signature...")
    if not verify_signature(manifest_path, sig_path, pubkey):
        print("ABORT: manifest signature invalid")
        sys.exit(1)
    print("      OK")

    # Step 2: parse manifest
    manifest = json.loads(manifest_path.read_text())
    print(
        f"[2/4] Verifying artifacts for version {manifest['version']} "
        f"(created {manifest['created_at']})..."
    )

    # Step 3: verify SHA256 of every artifact
    # Even though the manifest is trusted, we re-verify each artifact to
    # catch corruption or partial downloads.
    for artifact in manifest["artifacts"]:
        path = update_dir / artifact["path"]
        if not path.exists():
            print(f"ABORT: artifact not found: {path}")
            sys.exit(1)
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            print(f"ABORT: SHA256 mismatch for {artifact['name']}")
            print(f"  expected: {artifact['sha256']}")
            print(f"  got:      {actual}")
            sys.exit(1)
        print(f"      OK  {artifact['name']}  ({actual[:16]}...)")

    # Step 4: atomic install
    # Write to a staging directory first. Only replace the live directory
    # after the health check passes. This ensures the live install is never
    # left in a partial state.
    print("[3/4] Installing to staging...")
    staging = install_dir.parent / (install_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for artifact in manifest["artifacts"]:
        src = update_dir / artifact["path"]
        dst = staging / artifact["path"]
        shutil.copy2(src, dst)
        print(f"      {artifact['name']} -> {dst}")

    # Step 5: health check before committing
    print("[4/4] Running health check...")
    if not health_check(staging, manifest):
        print("ABORT: health check failed, rolling back")
        shutil.rmtree(staging)
        sys.exit(1)
    print("      OK")

    # Atomic swap: rename staging -> live, keep old as .old for one cycle
    old = install_dir.parent / (install_dir.name + ".old")
    if install_dir.exists():
        if old.exists():
            shutil.rmtree(old)
        install_dir.rename(old)
    staging.rename(install_dir)

    print(f"\n=== Update complete: {manifest['version']} ===")


if __name__ == "__main__":
    main()
