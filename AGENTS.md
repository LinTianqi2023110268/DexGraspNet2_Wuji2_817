# Long-term repository rules

- Do not modify the official Wuji2 URDF/USD to hide integration errors.
- Run cuRobo only in the `curobo_v2` conda environment.
- Run Isaac Lab only in the `isaaclab22_sim50` conda environment.
- Check `git status` and `git diff` before major changes.
- Record important commands in `core/worklog/COMMAND_LOG.md`.
- Record phase conclusions in `core/worklog/SESSION_SUMMARY.md`.
- Archive legacy implementations before removing them from the production path.
- Do not relax acceptance thresholds without explicit user approval.
