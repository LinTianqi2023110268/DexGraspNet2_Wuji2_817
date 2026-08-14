"""Canonical Wuji Hand 2 asset contract and legacy-pose compatibility.

Isaac Sim must load the official USD.  Offline FK and collision code uses the
companion URDF from the exact same upstream commit.  Existing Wuji2 1.0 labels
store ``T_world_r_base_link`` and therefore require the explicit conversion
below before they can drive the official ``r_wrist``-rooted articulation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "config/wuji2_official_asset.json"


def load_asset_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def canonical_asset_paths() -> tuple[Path, Path]:
    contract = load_asset_contract()["canonical_release"]
    return (
        resolve_project_path(contract["usd_entry"]),
        resolve_project_path(contract["companion_urdf"]),
    )


def canonical_usd_path() -> Path:
    """Return the official layered USD entry without touching any URDF file."""

    contract = load_asset_contract()["canonical_release"]
    return resolve_project_path(contract["usd_entry"])


def verify_canonical_usd() -> dict[str, str]:
    """Verify only the official USD used by Isaac Sim runtime.

    This deliberately does not open or hash the companion URDF.  It is the
    verifier used by the new official-USD grasp scripts, whose hand runtime
    contract has exactly one source of truth: the composed USD asset.
    """

    contract = load_asset_contract()["canonical_release"]
    usd = canonical_usd_path()
    if not usd.is_file():
        raise FileNotFoundError(usd)
    expected_files = {usd: contract["usd_sha256"]}
    expected_files.update(
        {
            usd.parent / relative: expected
            for relative, expected in contract["usd_layer_sha256"].items()
        }
    )
    actual: dict[str, str] = {}
    for path, expected in expected_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"Canonical Wuji2 USD layer changed: {path}; "
                f"expected {expected}, got {digest}"
            )
        actual[str(path)] = digest
    return actual


def verify_canonical_assets() -> dict[str, str]:
    contract = load_asset_contract()["canonical_release"]
    usd, urdf = canonical_asset_paths()
    expected = {
        usd: contract["usd_sha256"],
        urdf: contract["companion_urdf_sha256"],
    }
    actual: dict[str, str] = {}
    for path, expected_digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise RuntimeError(
                f"Canonical Wuji2 asset changed: {path}; "
                f"expected {expected_digest}, got {digest}"
            )
        actual[str(path)] = digest
    return actual


def legacy_base_pose_to_official_wrist(
    poses: np.ndarray,
) -> np.ndarray:
    """Convert one or more legacy ``T_world_r_base_link`` matrices.

    The 20 joint values remain semantic joint angles and are mapped separately
    by name.  This function changes only the hand-root coordinate convention.
    """

    source = np.asarray(poses)
    if source.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (...,4,4) poses, got {source.shape}")
    contract = load_asset_contract()["legacy_dataset_compatibility"]
    base_to_legacy_wrist = np.asarray(
        contract["legacy_base_T_legacy_wrist"], dtype=np.float64
    )
    official_to_legacy_wrist = np.asarray(
        contract["official_wrist_T_coordinate_to_legacy_wrist"],
        dtype=np.float64,
    )
    converted = (
        source.astype(np.float64)
        @ base_to_legacy_wrist
        @ official_to_legacy_wrist
    )
    return converted.astype(source.dtype, copy=False)


def poses_in_official_wrist_frame(
    poses: np.ndarray, source_frame: str
) -> np.ndarray:
    normalized = str(source_frame).strip().lower()
    if normalized in {"official_r_wrist", "r_wrist_hand2_beta1"}:
        return np.asarray(poses).copy()
    if normalized in {"legacy_r_base_link", "r_base_link"}:
        return legacy_base_pose_to_official_wrist(poses)
    raise ValueError(f"Unknown Wuji2 hand-root pose frame: {source_frame!r}")
