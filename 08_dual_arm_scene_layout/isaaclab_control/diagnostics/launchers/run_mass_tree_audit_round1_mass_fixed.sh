#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
CONTROL_ROOT="${PROJECT_ROOT}/08_dual_arm_scene_layout/isaaclab_control"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
ISAACLAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
PROGRAM="${CONTROL_ROOT}/diagnostics/scripts/04_audit_mass_tree.py"
CONFIG="${CONTROL_ROOT}/diagnostics/config/initial_stability_grouped_pd_round1_mass_fixed.json"
OUTPUT="${CONTROL_ROOT}/outputs/mass_tree_audit_round1_mass_fixed/report.json"
LOCK_FILE="/tmp/dgn2_wuji2_isaaclab_gpu.lock"

for required in "${CONDA_SETUP}" "${ISAAC_SIM_ENV}" "${ISAACLAB_ROOT}/isaaclab.sh" "${PROGRAM}" "${CONFIG}"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "[GPU PREFLIGHT REFUSED] Another guarded Isaac task is running." >&2; exit 20; }
[[ -e /dev/nvidia0 && -e /dev/nvidiactl && -e /dev/nvidia-uvm ]] || {
    echo "[GPU PREFLIGHT REFUSED] NVIDIA device nodes are incomplete." >&2; exit 21;
}
nvidia-smi >/dev/null 2>&1 || {
    echo "[GPU PREFLIGHT REFUSED] nvidia-smi cannot communicate with the driver." >&2; exit 22;
}
if journalctl -k -b --no-pager 2>/dev/null | grep -Eq 'NVRM: Xid .*: (31|154),'; then
    echo "[GPU PREFLIGHT REFUSED] Current boot contains NVIDIA Xid 31/154." >&2
    exit 23
fi
existing_isaac="$({ pgrep -a -f '/kit/kit|isaac-sim\.sh|[p]ython .*04_audit_mass_tree\.py' || true; })"
if [[ -n "${existing_isaac}" ]]; then
    echo "[GPU PREFLIGHT REFUSED] An Isaac/Kit process already exists:" >&2
    echo "${existing_isaac}" >&2
    exit 24
fi

echo "[GPU PREFLIGHT PASS] Starting one-step mass/COM/gravity audit."
export TERM=xterm-256color PYTHONUNBUFFERED=1
source "${CONDA_SETUP}"
conda activate isaaclab22_sim50
set +u
source "${ISAAC_SIM_ENV}"
set -u
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_tasks${PYTHONPATH:+:${PYTHONPATH}}"

exec systemd-inhibit \
    --what=sleep:handle-lid-switch \
    --who="DexGraspNet2_Wuji2 mass audit" \
    --why="Prevent CUDA/PhysX context loss during the one-step audit" \
    --mode=block \
    bash "${ISAACLAB_ROOT}/isaaclab.sh" -p "${PROGRAM}" \
        --config "${CONFIG}" --output "${OUTPUT}" "$@"
