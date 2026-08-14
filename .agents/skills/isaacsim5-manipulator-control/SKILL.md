---
name: isaacsim5-manipulator-control
description: Audits and configures Isaac Sim 5.0 PhysX manipulator assets, joint drives, gains, limits, mass properties, gravity holding, and static-stability tests. Use for this repository's dual-arm + Wuji2 physics/control debugging, URDF/USD import questions, Force vs Acceleration Drive, PD tuning, gravity torque, payload, COM, or startup instability.
audience: user
status: experimental
owners:
  - project-maintainer
---

# When To Use

Use this skill whenever work touches the Isaac Sim 5.0 physics/control layer of the repository's own dual-arm + Wuji2 robot, especially:

- URDF/USD import or composition;
- articulation root, joint drive, stiffness/damping, effort or velocity limits;
- Force Drive versus Acceleration Drive;
- startup shaking, sagging, drift, gravity holding, torque saturation;
- payload, mass, inertia, center of mass, duplicate end-effector mass;
- static-stability acceptance before IK.

Do **not** use UR10/UR10e/Franka as a replacement robot. NVIDIA manipulators are reference implementations only. The final system must keep the project's own dual-arm model and Wuji2.

# Workflow

1. **Lock the target stack before changing code.**
   - Target: Isaac Sim 5.0, PhysX, Isaac Lab 2.2.x.
   - Do not introduce Isaac Sim 6.x or Isaac Lab 3.x APIs without an explicit migration request.
   - Before relying on an API, search the local Isaac Sim 5.0 installation/source/examples first.

2. **Preserve source-of-truth robot parameters.**
   - Keep the project's own URDF/USD kinematics, mass, inertia, COM, joint limits, effort limits, and velocity limits unless the user explicitly asks to change them.
   - Right-arm hard effort limits currently expected from the project URDF: J1 130 N·m; J2-J4 70 N·m; J5-J7 12 N·m.
   - Do not raise the 12 N·m wrist limit merely to make a test pass.
   - Do not overwrite Wuji2 official drive parameters when adjusting the arm.

3. **Audit before tuning.** For every right-arm joint, report the runtime:
   - drive type;
   - target type;
   - stiffness;
   - damping;
   - maxForce / simulation effort limit;
   - velocity limit;
   - armature/friction if authored;
   - current target and current state.

4. **Treat Force and Acceleration Drive as different actuator models.**
   - Never compare or copy stiffness/damping values across the two modes as if they were the same torque-PD gains.
   - Isaac Sim 5.0 documents Force Drive as direct spring-damper effort and Acceleration Drive as inertia-normalized drive.
   - If changing drive type is proposed, first identify where the current drive type was authored: URDF importer, USD layer, composition, or Isaac Lab override.

5. **Static stability is a hard gate.**
   - Gravity ON.
   - IK OFF.
   - No camera task, trajectory, grasp, or state-machine motion.
   - Explicitly hold the initial target pose.
   - Validate right arm, Wuji2, flange, and wrist separately.
   - If static stability FAILS, do not proceed to IK.

6. **Diagnose gravity failures with evidence, not blind gain changes.**
   - A zero-gravity PASS with gravity-ON FAIL narrows the issue to gravity/load/actuation, but does not by itself prove a particular joint is under-rated.
   - If using `get_gravity_compensation_forces()`, label it as the PhysX gravity compensation term G(q) for the current simulated model.
   - Do not substitute projected joint force for gravity compensation torque.
   - Before concluding a joint cannot hold the pose, audit all downstream rigid-body masses, COMs and inertias, and confirm no old gripper/finger/camera payload is duplicated with Wuji2.

7. **Tune only after the model and actuator mode are known.**
   - Change one variable at a time.
   - Sag without oscillation: investigate gravity demand, saturation, payload/COM, then stiffness.
   - Oscillation around target: investigate damping versus stiffness and solver settings.
   - Immediate large jump: stop tuning; re-check state/target mapping, drive target, limits, collisions, and authored runtime state.
   - Do not copy UR10e gain numbers directly to this robot.

8. **Use NVIDIA official manipulators only as controls/baselines.**
   - Compare configuration patterns: articulation, drive type, gains source, effort limits, solver iterations, reset/initialization, IK organization.
   - Keep the project's own robot geometry and dynamics.

# Validation

A physics/control change is not complete until it produces an auditable report containing at least:

- exact robot/USD source loaded;
- articulation root and expected joint count;
- right-arm joint names and runtime drive properties;
- gravity status;
- per-joint max drift and velocity during the static test;
- flange and Wuji2-wrist pose drift;
- NaN, joint-limit, collision, and sustained-oscillation flags;
- separate PASS/FAIL for right arm and Wuji2;
- overall `STATIC STABILITY: PASS/FAIL`.

For the current project, `STATIC STABILITY: PASS` is required before enabling Differential IK or the 20 mm short-motion test.

# Maintenance

When this skill conflicts with current local code, prefer in this order:

1. local Isaac Sim 5.0 source and runtime USD attributes;
2. Isaac Sim 5.0 official docs/examples;
3. local project URDF/USD and measured runtime properties;
4. newer Isaac Sim documentation only for concepts that are verified unchanged.

Re-check this skill if the repository migrates away from Isaac Sim 5.0, changes the robot asset, changes the Wuji2 mount/payload, or changes actuator mode.

# References

Read `reference.md` for version-specific notes and official documentation links.
Read `evaluations.md` before modifying the skill itself.
