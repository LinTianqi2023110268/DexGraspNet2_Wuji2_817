#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control"
CONFIG_DIR="${ROOT}/history/config/accel_nf_scan"

python3 "${ROOT}/history/scripts/05_make_accel_nf_scan_configs.py"

for config in "${CONFIG_DIR}"/*.json; do
    echo
    echo "[ACCEL NF SCAN] ${config}"
    set +e
    bash "${ROOT}/diagnostics/launchers/run_initial_stability.sh" --config "${config}" "$@"
    status=$?
    set -e
    output_dir="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_directory"])' "${config}")"
    report="${ROOT}/../../${output_dir}/report.json"
    report_status="MISSING"
    if [[ -f "${report}" ]]; then
        report_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${report}")"
    fi
    if [[ "${report_status}" == "PASS" ]]; then
        echo "[ACCEL NF SCAN PASS] ${config}"
        exit 0
    fi
    if [[ ${status} -ne 2 ]]; then
        echo "[ACCEL NF SCAN STOP] unexpected status=${status}, report=${report_status} for ${config}" >&2
        exit "${status}"
    fi
    echo "[ACCEL NF SCAN FAIL] report=${report_status}; continuing to next natural frequency"
done

echo "[ACCEL NF SCAN DONE] no PASS in prepared scan range" >&2
exit 2
