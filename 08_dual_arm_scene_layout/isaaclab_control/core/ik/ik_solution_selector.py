from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .curobo_gpu_ik import BatchedIKResult


@dataclass(frozen=True)
class SelectedIK:
    target_index: int
    solution_index: int
    q_rad: np.ndarray
    q_reference_rad: np.ndarray
    normalized_joint_distance: float
    position_error_m: float
    orientation_error_rad: float
    inner_limit_margin_rad: float

    def to_jsonable(self) -> dict:
        return {
            "target_index": int(self.target_index),
            "solution_index": int(self.solution_index),
            "q_rad": self.q_rad.tolist(),
            "q_reference_rad": self.q_reference_rad.tolist(),
            "normalized_joint_distance": float(self.normalized_joint_distance),
            "position_error_m": float(self.position_error_m),
            "orientation_error_rad": float(self.orientation_error_rad),
            "inner_limit_margin_rad": float(self.inner_limit_margin_rad),
        }


def normalized_joint_distance(q: np.ndarray, q_ref: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.maximum(np.asarray(hi) - np.asarray(lo), 1.0e-9)
    delta = (np.asarray(q) - np.asarray(q_ref)) / span
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _unique_indices(q: np.ndarray, decimals: int) -> np.ndarray:
    rounded = np.round(np.asarray(q), decimals=decimals)
    _, first = np.unique(rounded, axis=0, return_index=True)
    return np.sort(first)


def select_solution(
    result: BatchedIKResult,
    target_index: int,
    q_reference_rad: np.ndarray,
    dedupe_round_decimals: int = 3,
) -> SelectedIK | None:
    """Select one accepted IK branch, prioritizing continuity from q_reference.

    Hard gates have already been applied in ``result.accepted``.  Among accepted
    solutions the lexicographic order is:
      1) smallest normalized joint-space distance to the current/reference pose,
      2) largest inner joint-limit margin,
      3) smallest normalized pose error.
    """
    i = int(target_index)
    q_ref = np.asarray(q_reference_rad, dtype=np.float64).reshape(7)
    accepted_idx = np.flatnonzero(result.accepted[i])
    if accepted_idx.size == 0:
        return None
    q_acc = result.q_rad[i, accepted_idx]
    keep_local = _unique_indices(q_acc, dedupe_round_decimals)
    accepted_idx = accepted_idx[keep_local]
    q_acc = result.q_rad[i, accepted_idx]
    dist = normalized_joint_distance(q_acc, q_ref, result.lower_inner_rad, result.upper_inner_rad)
    margin = result.inner_limit_margin_rad[i, accepted_idx]
    pos_norm = result.position_error_m[i, accepted_idx]
    rot_norm = result.orientation_error_rad[i, accepted_idx]
    # np.lexsort uses the last key as primary.
    order = np.lexsort((pos_norm + rot_norm, -margin, dist))
    k_local = int(order[0])
    k = int(accepted_idx[k_local])
    return SelectedIK(
        target_index=i,
        solution_index=k,
        q_rad=result.q_rad[i, k].copy(),
        q_reference_rad=q_ref.copy(),
        normalized_joint_distance=float(dist[k_local]),
        position_error_m=float(result.position_error_m[i, k]),
        orientation_error_rad=float(result.orientation_error_rad[i, k]),
        inner_limit_margin_rad=float(result.inner_limit_margin_rad[i, k]),
    )


def select_waypoint_chain(
    result: BatchedIKResult,
    initial_reference_rad: np.ndarray,
    dedupe_round_decimals: int = 3,
) -> list[SelectedIK] | None:
    """Select a continuous IK branch through an ordered waypoint batch.

    The first waypoint is anchored to the actual current/initial arm pose.  Every
    subsequent waypoint is anchored to the solution selected for its predecessor.
    """
    q_ref = np.asarray(initial_reference_rad, dtype=np.float64).reshape(7)
    selected: list[SelectedIK] = []
    for i in range(result.batch_size):
        pick = select_solution(result, i, q_ref, dedupe_round_decimals)
        if pick is None:
            return None
        selected.append(pick)
        q_ref = pick.q_rad
    return selected
