#!/usr/bin/env python3
"""依次完成5个单视角的网络预测和场景碰撞过滤。

输入：
  config/test_5scene.json：5个scene/view选择。
  正式测试集network_input.npz：每个视角40000点。
  config/project.json中的默认50000步Wuji2 checkpoint。

输出（每个scene/view一组）：
  raw_predictions.npz/json：1024条网络原始候选和网络诊断。
  filtered_predictions.npz/json：通过场景、桌面和虎口路径过滤的候选。
  filtered_predictions_all_diagnostics.npz：所有被测候选的过滤原因。
  balanced_raw_predictions.npz/json：官方categorical模式按真实分割ID均衡采样。
  balanced_filtered_predictions.npz/json：均衡候选的相同碰撞过滤结果。

raw/filtered是无需目标ID的正常部署推理；balanced_*只用于本次合成测试集中
“每个物体展示两条位姿”的逐物体诊断，它读取仿真真值seg，不能冒充真实部署。

本文件只负责顺序调用正式05入口，不在这里重新实现网络和碰撞公式。
已存在的完整输出默认跳过；加入--overwrite才覆盖。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = PROJECT_ROOT / (
    "02_training_dataset/data/scene_datasets/"
    "wuji2_test60_10upright_10view_v1"
)
SELECTION = PROJECT_ROOT / "07_wuji2_network_3p3r_sim/00_config/test_5scene.json"
RUNTIME_CONFIG = (
    PROJECT_ROOT / "07_wuji2_network_3p3r_sim/00_config/test_runtime_config.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "07_wuji2_network_3p3r_sim/01_cases"
PREDICT = PROJECT_ROOT / "05_inference/01_predict_single_view.py"
FILTER = PROJECT_ROOT / "05_inference/02_filter_scene_collisions.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grasp-num", type=int, default=1024)
    parser.add_argument("--collision-batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    progress = []
    for case in selection["scenes"]:
        scene = int(case["scene_index"])
        view = int(case["view_index"])
        case_root = (
            OUTPUT_ROOT
            / f"scene_{scene:04d}_view_{view:04d}"
            / "02_predictions"
        )
        raw = case_root / "raw_predictions.npz"
        filtered = case_root / "filtered_predictions.npz"
        balanced_raw = case_root / "balanced_raw_predictions.npz"
        balanced_filtered = case_root / "balanced_filtered_predictions.npz"

        if args.overwrite or not (raw.is_file() and raw.with_suffix(".json").is_file()):
            run(
                [
                    sys.executable,
                    str(PREDICT),
                    "--data-root", str(TEST_ROOT),
                    "--scene", str(scene),
                    "--view", str(view),
                    "--output", str(raw),
                    "--device", args.device,
                    "--grasp-num", str(args.grasp_num),
                ]
            )
        else:
            print(f"[SKIP] raw scene={scene:04d} view={view:04d}")

        if args.overwrite or not (
            filtered.is_file()
            and filtered.with_suffix(".json").is_file()
            and filtered.with_name(filtered.stem + "_all_diagnostics.npz").is_file()
        ):
            run(
                [
                    sys.executable,
                    str(FILTER),
                    "--config", str(RUNTIME_CONFIG),
                    "--prediction", str(raw),
                    "--output", str(filtered),
                    "--device", args.device,
                    "--collision-batch-size", str(args.collision_batch_size),
                ]
            )
        else:
            print(f"[SKIP] filtered scene={scene:04d} view={view:04d}")

        if args.overwrite or not (
            balanced_raw.is_file() and balanced_raw.with_suffix(".json").is_file()
        ):
            run(
                [
                    sys.executable,
                    str(PREDICT),
                    "--data-root", str(TEST_ROOT),
                    "--scene", str(scene),
                    "--view", str(view),
                    "--output", str(balanced_raw),
                    "--device", args.device,
                    "--grasp-num", str(args.grasp_num),
                    "--categorical",
                ]
            )
        else:
            print(f"[SKIP] balanced raw scene={scene:04d} view={view:04d}")

        if args.overwrite or not (
            balanced_filtered.is_file()
            and balanced_filtered.with_suffix(".json").is_file()
            and balanced_filtered.with_name(
                balanced_filtered.stem + "_all_diagnostics.npz"
            ).is_file()
        ):
            run(
                [
                    sys.executable,
                    str(FILTER),
                    "--config", str(RUNTIME_CONFIG),
                    "--prediction", str(balanced_raw),
                    "--output", str(balanced_filtered),
                    "--device", args.device,
                    "--collision-batch-size", str(args.collision_batch_size),
                ]
            )
        else:
            print(f"[SKIP] balanced filtered scene={scene:04d} view={view:04d}")

        filter_summary = json.loads(filtered.with_suffix(".json").read_text())
        balanced_filter_summary = json.loads(
            balanced_filtered.with_suffix(".json").read_text()
        )
        progress.append(
            {
                "scene_index": scene,
                "view_index": view,
                "raw_prediction": raw.relative_to(PROJECT_ROOT).as_posix(),
                "filtered_prediction": filtered.relative_to(PROJECT_ROOT).as_posix(),
                "raw_candidate_count": int(filter_summary["input_candidates"]),
                "filtered_candidate_count": int(filter_summary["enhanced_final_kept"]),
                "balanced_raw_prediction": balanced_raw.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "balanced_filtered_prediction": balanced_filtered.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "balanced_filtered_candidate_count": int(
                    balanced_filter_summary["enhanced_final_kept"]
                ),
            }
        )

    report = {
        "schema_version": 1,
        "status": "five_scene_network_and_filter_complete",
        "device_requested": args.device,
        "cases": progress,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_ROOT / "network_and_filter_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[COMPLETE] {report_path}")


if __name__ == "__main__":
    main()
