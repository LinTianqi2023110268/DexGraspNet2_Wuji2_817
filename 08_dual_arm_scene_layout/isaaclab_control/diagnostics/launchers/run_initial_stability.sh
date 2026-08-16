#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
CONDA_SETUP="/home/lin/miniconda3/etc/profile.d/conda.sh"
ISAAC_SIM_ENV="/home/lin/isaacsim/setup_conda_env.sh"
ISAACLAB_ROOT="/home/lin/Projects/Wuji2_DexGraspNet_Portable_Dataset_Factory/01_environment/vendor/IsaacLab-2.2.0"
PROGRAM="${PROJECT_ROOT}/08_dual_arm_scene_layout/isaaclab_control/diagnostics/scripts/00_check_initial_stability.py"
LOCK_FILE="/tmp/dgn2_wuji2_isaaclab_gpu.lock"

for required in "${CONDA_SETUP}" "${ISAAC_SIM_ENV}" "${ISAACLAB_ROOT}/isaaclab.sh" "${PROGRAM}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required file: ${required}" >&2
        exit 1
    fi
done

# One Isaac/PhysX GPU process at a time on this 8 GiB workstation.  A second
# process can invalidate CUDA/Vulkan interop and leave Kit printing error 999.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "[GPU PREFLIGHT REFUSED] Another guarded Isaac Lab task is running." >&2
    echo "Close it first; this test will not start a second Kit process." >&2
    exit 20
fi

if [[ ! -e /dev/nvidia0 || ! -e /dev/nvidiactl || ! -e /dev/nvidia-uvm ]]; then
    echo "[GPU PREFLIGHT REFUSED] NVIDIA device nodes are incomplete." >&2
    echo "Reboot before starting Isaac Lab; do not reinstall the driver." >&2
    exit 21
fi

if ! nvidia-smi >/dev/null 2>&1; then
    echo "[GPU PREFLIGHT REFUSED] nvidia-smi cannot communicate with the driver." >&2
    echo "Reboot before starting Isaac Lab; do not launch Kit in this state." >&2
    exit 22
fi

# Xid 31 followed by Xid 154 means the current GPU context requires a node
# reboot.  Starting Kit after that only creates an endless cudaErrorUnknown 999
# loop.  Restrict this check to the current boot so historical failures do not
# block a recovered machine.
if journalctl -k -b --no-pager 2>/dev/null | grep -Eq 'NVRM: Xid .*: (31|154),'; then
    echo "[GPU PREFLIGHT REFUSED] Current boot contains NVIDIA Xid 31/154." >&2
    echo "The driver marked GPU recovery as reboot-required. Reboot first." >&2
    exit 23
fi

existing_isaac="$({ pgrep -a -f '/kit/kit|isaac-sim\.sh|[p]ython .*00_check_initial_stability\.py' || true; })"
if [[ -n "${existing_isaac}" ]]; then
    echo "[GPU PREFLIGHT REFUSED] An Isaac/Kit process already exists:" >&2
    echo "${existing_isaac}" >&2
    echo "Close that process before launching this static test." >&2
    exit 24
fi

echo "[GPU PREFLIGHT PASS] NVIDIA driver/device nodes are healthy."
echo "[GPU PREFLIGHT PASS] No current-boot Xid 31/154 and no other Isaac process."
echo "[SLEEP INHIBITOR] Suspend and lid-switch sleep are blocked only while this test runs."

export TERM=xterm-256color PYTHONUNBUFFERED=1
source "${CONDA_SETUP}"
conda activate isaaclab22_sim50

# setup_conda_env.sh probes optional shell variables. Keep nounset disabled
# only while sourcing it; command failures remain fatal throughout.
set +u
source "${ISAAC_SIM_ENV}"
set -u
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_tasks${PYTHONPATH:+:${PYTHONPATH}}"

exec systemd-inhibit \
    --what=sleep:handle-lid-switch \
    --who="DexGraspNet2_Wuji2 Isaac Lab" \
    --why="Prevent NVIDIA CUDA/PhysX context loss during the static test" \
    --mode=block \
    bash "${ISAACLAB_ROOT}/isaaclab.sh" -p "${PROGRAM}" "$@"
