from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CollisionPhase(str, Enum):
    PREGRASP = "pregrasp"
    APPROACH = "approach"
    COVER = "cover"
    GRASP = "grasp"
    SQUEEZE = "squeeze"
    LIFT = "lift"


@dataclass(frozen=True)
class PhaseCollisionPolicy:
    """First-version target contact semantics.

    Non-target observed scene geometry is always an obstacle.  Target geometry is an
    obstacle before contact and is allowed during grasp/squeeze/lift.  Once Wuji2 link
    groups are wired in, this broad target allowance must be narrowed to intentional
    finger-contact links only; the API is intentionally phase-aware so that change does
    not require replacing the mapper.
    """

    phase: CollisionPhase

    @property
    def target_is_obstacle(self) -> bool:
        return self.phase in {
            CollisionPhase.PREGRASP,
            CollisionPhase.APPROACH,
            CollisionPhase.COVER,
        }
