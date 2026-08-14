---
name: isaaclab22-manipulator-control
description: Implements and reviews Isaac Lab 2.2.x manipulator control loops for this repository using AppLauncher, SimulationContext, Articulation, scene state, Differential IK, safe initialization, command-line telemetry, and staged validation. Use for the dual-arm + Wuji2 Isaac Lab control subproject, 35-DOF articulation checks, right-arm 7-joint control, flange tracking, Jacobians, short-motion tests, or headless launch/debugging.
audience: user
status: experimental
owners:
  - project-maintainer
---

# When To Use

Use this skill for Isaac Lab 2.2.x application structure and robot-control code in the repository, including:

- starting Isaac Sim through Isaac Lab;
- scene/articulation discovery and self-checks;
- initializing and holding the 35-DOF combined articulation;
- reading right-arm 7-joint and flange state;
- Differential IK and Jacobian handling;
- short smooth flange motion and return;
- terminal progress/telemetry and headless tests.

Do not use this skill to introduce RL. This project's current control milestone is deterministic manipulator control.

# Workflow

1. **Respect the installed version and local source.**
   - Target: Isaac Lab 2.2.x with Isaac Sim 5.0 / PhysX.
   - Do not use Isaac Lab 3.x multi-backend APIs or Isaac Sim 6.x APIs unless explicitly requested.
   - Search the local `IsaacLab-2.2.0` source/tutorials before using an uncertain API.

2. **Use the Isaac Lab application lifecycle.**
   - `AppLauncher` owns Isaac Sim startup/shutdown.
   - `SimulationContext` owns stepping/reset timing.
   - `Articulation` owns robot state and commands.
   - Use `InteractiveScene` where it already fits the repository's scene organization.
   - Do not run Isaac Lab inside an already-open Isaac Sim Script Editor session.
   - Do not start a second GUI/Kit instance from the same control script.

3. **Preserve the calibrated scene.**
   - Do not rewrite existing calibrated scene/camera/network-preprocessing files just to run control tests.
   - Add control code under the independent `isaaclab_control` area or a derived control stage/config.

4. **Fail closed during self-check.** Before applying any command, verify:
   - the expected articulation is uniquely resolved;
   - the expected total joint count is present (currently 35 in the calibrated setup; verify runtime);
   - all seven right-arm joints are found in the correct order;
   - the right flange body is found;
   - Wuji2 joints/wrist are resolved when needed;
   - fixed-base status and relevant body/joint indices are recorded.
   - If any check fails, stop without driving the robot.

5. **Initialize the entire articulation explicitly.**
   - After simulation reset, write the intended joint positions for the whole articulation and zero joint velocities.
   - Reset articulation buffers as required by the local Isaac Lab 2.2 example/API.
   - Set position targets to the same initial state before the static hold.
   - Do not initialize only the right arm while leaving other DOFs with stale targets.

6. **Static stability is the hard gate.**
   - Gravity ON, IK OFF, no trajectory, no grasp, no camera task.
   - Warm-up may be used, but any violent motion during warm-up is a FAIL.
   - Record a formal 10-second stability window.
   - Right arm and Wuji2 receive separate PASS/FAIL results.
   - If overall static stability is not PASS, Differential IK remains locked.

7. **Only after PASS, enable the minimal short-motion test.**
   - Read current right-arm joint state and flange pose.
   - Generate a short, smooth Cartesian trajectory (the project's current target is a small 20 mm-class move and return; use config values, not magic numbers).
   - Use a fifth-order time-scaling/curve if that is the configured test requirement.
   - Keep Wuji2 hand commands fixed during the arm-only motion unless the test explicitly targets the hand.

8. **Use Differential IK according to local Isaac Lab 2.2 examples.**
   - Controller input: desired end-effector pose, current joint positions, current end-effector pose, and the correct Jacobian slice.
   - For fixed-base articulations, verify the flange body index versus Jacobian body index using the local `run_diff_ik.py`; do not assume they are identical.
   - Do not apply an extra world-to-root Jacobian rotation if the local PhysX/Isaac Lab API already returns the Jacobian in articulation-root coordinates.
   - Keep only the right-arm active joint columns in the arm IK solve; do not let Wuji2 finger joints enter the arm Jacobian solve.

9. **Keep the simulation loop flat and readable.**
   Prefer the sequence:

   `read state -> update state machine/desired pose -> controller compute -> set targets -> write data to sim -> sim.step -> scene/robot update -> telemetry`

   Split configuration, scene loading, self-check, trajectory generation, control loop, and reporting into small modules/functions. Avoid deep nested conditionals and hidden magic numbers.

10. **Command-line telemetry is mandatory for long tests.**
    - Print state transitions as new log lines.
    - Update live position/error/progress on one terminal line around 5 Hz instead of printing every physics step.
    - Log max joint velocity, flange error, state, timeout, NaN, and limit events.
    - Always write a machine-readable report and trace CSV for acceptance tests.

# Validation

For the current project, validation proceeds in this order:

1. import/startup check;
2. articulation self-check;
3. static stability test;
4. only after static PASS: right-flange short smooth move;
5. smooth return to start;
6. report maximum flange error and maximum joint speed;
7. only after the minimal arm loop passes: camera/perception/grasp stages may be connected.

A run must fail safely if the articulation/joints/flange cannot be resolved or if static stability fails.

# Maintenance

Authoritative sources, in order:

1. local Isaac Lab 2.2.x source and tutorials;
2. Isaac Lab 2.2.x official docs;
3. repository's working launch wrappers/configuration;
4. newer Isaac Lab docs only for concepts verified to be unchanged.

Review this skill if the project upgrades Isaac Lab, changes the combined articulation, changes the flange name, or moves from Differential IK to another controller.

# References

Read `reference.md` for project-specific launch and IK notes.
Read `evaluations.md` before modifying this skill.
