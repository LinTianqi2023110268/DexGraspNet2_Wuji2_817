"""Deterministic Wuji2 pinky coupling used after four-finger retargeting.

LEAP provides only a thumb plus three opposing fingers.  The official
four-finger solve therefore leaves Wuji2's fifth finger inactive.  This module
adds the user-approved policy without changing the official optimizer:

* copy the ring finger's flexion chain to the pinky;
* copy ring MCP abduction, then add a small outward bias;
* respect the official Wuji2 URDF limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config/pinky_ring_coupling.json"


def load_pinky_policy(path: Path = DEFAULT_CONFIG) -> dict:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("policy") != "copy_ring_flexion_and_bias_abduction_outward":
        raise ValueError(f"unsupported pinky policy: {policy.get('policy')}")
    if policy.get("limit_policy") != "clip_to_official_urdf_limits":
        raise ValueError(f"unsupported pinky limit policy: {policy.get('limit_policy')}")
    return policy


def apply_pinky_ring_coupling(
    qpos: np.ndarray,
    joint_names: list[str],
    joint_limits: np.ndarray,
    policy: dict,
) -> tuple[np.ndarray, dict]:
    """Return qpos with pinky flexion copied from ring and outward side bias.

    ``qpos`` may be one q20 vector or an array ending in q20.  Only the four
    pinky columns are changed.  Abduction is clipped only when the requested
    outward offset reaches an official URDF limit; the effective offset is
    included in the returned audit dictionary.
    """

    output = np.asarray(qpos, dtype=np.float64).copy()
    limits = np.asarray(joint_limits, dtype=np.float64)
    if output.shape[-1] != len(joint_names):
        raise ValueError(f"qpos shape {output.shape} does not match {len(joint_names)} joints")
    if limits.shape != (len(joint_names), 2):
        raise ValueError(f"joint_limits must be {(len(joint_names), 2)}, got {limits.shape}")

    index = {name: position for position, name in enumerate(joint_names)}
    mapping = dict(policy["joint_mapping"])
    for target, source in mapping.items():
        output[..., index[target]] = output[..., index[source]]

    abd = policy["abduction_mapping"]
    target_index = index[str(abd["target"])]
    source_index = index[str(abd["source"])]
    requested_offset = (
        float(policy["outward_abduction_sign"])
        * np.deg2rad(float(policy["outward_abduction_offset_deg"]))
    )
    requested = output[..., source_index] + requested_offset
    output[..., target_index] = np.clip(
        requested,
        limits[target_index, 0],
        limits[target_index, 1],
    )

    if np.any(output < limits[:, 0] - 1.0e-7) or np.any(
        output > limits[:, 1] + 1.0e-7
    ):
        raise RuntimeError("pinky coupling produced a joint outside official limits")
    effective = output[..., target_index] - output[..., source_index]
    audit = {
        "policy": str(policy["policy"]),
        "requested_outward_abduction_offset_deg": float(
            policy["outward_abduction_offset_deg"]
        ),
        "requested_outward_abduction_offset_rad": float(requested_offset),
        "effective_outward_abduction_offset_rad_min": float(np.min(effective)),
        "effective_outward_abduction_offset_rad_max": float(np.max(effective)),
        "abduction_was_limit_clipped": bool(
            np.any(np.abs(output[..., target_index] - requested) > 1.0e-10)
        ),
    }
    return output, audit
