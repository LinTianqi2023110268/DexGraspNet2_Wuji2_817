from __future__ import annotations

"""Stage-specific post-solve IK acceptance.

Exact COVER does not use this module: it keeps the existing cuRobo/default strict
acceptance contract unchanged.

Relaxed non-contact stages reuse the returned cuRobo seeds and classify them from
their measured pose residuals and inner joint-limit margin. This deliberately
does not require cuRobo's strict ``raw_success`` flag unless the stage policy says so.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StageAcceptancePolicy:
    name: str
    position_tolerance_m: float
    orientation_tolerance_rad: float
    minimum_inner_limit_margin_rad: float
    require_raw_success: bool = False

    @classmethod
    def from_payload(cls, payload: dict) -> "StageAcceptancePolicy":
        return cls(
            name=str(payload.get("name", "unnamed")),
            position_tolerance_m=float(payload["position_tolerance_m"]),
            orientation_tolerance_rad=float(payload["orientation_tolerance_rad"]),
            minimum_inner_limit_margin_rad=float(
                payload["minimum_inner_limit_margin_rad"]
            ),
            require_raw_success=bool(payload.get("require_raw_success", False)),
        )

    def to_jsonable(self) -> dict:
        return {
            "name": self.name,
            "position_tolerance_m": float(self.position_tolerance_m),
            "orientation_tolerance_rad": float(self.orientation_tolerance_rad),
            "orientation_tolerance_deg": float(
                math.degrees(self.orientation_tolerance_rad)
            ),
            "minimum_inner_limit_margin_rad": float(
                self.minimum_inner_limit_margin_rad
            ),
            "minimum_inner_limit_margin_deg": float(
                math.degrees(self.minimum_inner_limit_margin_rad)
            ),
            "require_raw_success": bool(self.require_raw_success),
        }


def acceptance_mask_from_result(result, policy: StageAcceptancePolicy) -> np.ndarray:
    """Return [B,R] acceptance mask from actual returned-seed residuals."""
    q = np.asarray(result.q_rad, dtype=np.float64)
    raw = np.asarray(result.raw_success, dtype=bool)
    pos = np.asarray(result.position_error_m, dtype=np.float64)
    rot = np.asarray(result.orientation_error_rad, dtype=np.float64)
    margin = np.asarray(result.inner_limit_margin_rad, dtype=np.float64)

    finite = (
        np.isfinite(q).all(axis=-1)
        & np.isfinite(pos)
        & np.isfinite(rot)
        & np.isfinite(margin)
    )
    accepted = (
        finite
        & (pos <= policy.position_tolerance_m)
        & (rot <= policy.orientation_tolerance_rad)
        & (margin >= policy.minimum_inner_limit_margin_rad)
    )
    if policy.require_raw_success:
        accepted &= raw
    return accepted
