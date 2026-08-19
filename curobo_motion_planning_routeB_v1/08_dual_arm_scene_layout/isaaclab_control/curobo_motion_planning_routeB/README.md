# cuRobo Motion Planning Route B v1

Purpose:
Add a second motion route while keeping the original route untouched.

Legacy Route A:
current q -> keypoint IK -> quintic interpolation -> Isaac execution

Route B:
current q
 -> cuRobo planning with robot-cleaned depth ESDF
 -> PREGRASP

PREGRASP:
- existing relaxed IK source code is reused

COVER:
- existing strict IK source code is reused

After grasp:
GRASP/SQUEEZE remains unchanged

Then:
COVER grasp state
 -> attached object scene
 -> cuRobo LIFT/TRANSFER/PLACE planning
 -> RELEASE

This package only provides interfaces.
Codex adapts local project APIs.
