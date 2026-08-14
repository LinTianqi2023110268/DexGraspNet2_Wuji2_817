#!/usr/bin/env python3
"""从过滤结果中为每个可见物体选择两条抓取位姿。

输入：
  outputs/scene_xxxx/view_xxxx/balanced_raw_predictions.npz。
  outputs/scene_xxxx/view_xxxx/balanced_filtered_predictions_all_diagnostics.npz。
  config/test_5scene.json中的场景物体ID。

输出：
  selected_top2.npz：保留的候选数组。
  selected_top2.json：每个物体的得分、来源索引与姿态差异说明。
  outputs/selected_pose_catalog.csv：60条位姿的单行索引与净空摘要。

balanced输入由官方categorical模式生成，使用合成测试集真实seg在六个物体间
均衡分配种子；它用于逐物体诊断，不是无需标签的真实部署推理。

过滤层级：某物体若有至少2条Wuji2增强过滤候选，就只从增强层选择；否则
回退到作者训练标签采用的strict pregrasp场景/桌面净空层。回退会在JSON和NPZ
中明确标记，不会伪装成增强过滤通过。

选择规则：第一条取该物体最高分；第二条优先要求与第一条平移至少15 mm，
或旋转至少20度。若没有满足差异阈值的候选但仍有第二条，则保留次高分并在
JSON中明确标记diversity_relaxed=true；不会复制同一条候选。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION = PROJECT_ROOT / "07_wuji2_network_3p3r_sim/00_config/test_5scene.json"
OUTPUT_ROOT = PROJECT_ROOT / "07_wuji2_network_3p3r_sim/01_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-diversity-m", type=float, default=0.015)
    parser.add_argument("--rotation-diversity-deg", type=float, default=20.0)
    return parser.parse_args()


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def candidate_subset(
    source: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    total = int(source["qpos"].shape[0])
    result = {}
    for key, value in source.items():
        if value.ndim > 0 and value.shape[0] == total:
            result[key] = value[indices]
        else:
            result[key] = value
    return result


def merge_raw_and_diagnostics(
    raw: dict[str, np.ndarray], diagnostics: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """按诊断文件的得分排序重排原始候选，并附加全部过滤量。"""

    source_indices = np.asarray(
        diagnostics["source_candidate_index"], dtype=np.int64
    )
    merged = candidate_subset(raw, source_indices)
    merged["source_candidate_index"] = source_indices
    for key, value in diagnostics.items():
        if key == "source_candidate_index":
            continue
        merged[key] = value
    final_pose = np.repeat(
        np.eye(4, dtype=np.float32)[None], len(source_indices), axis=0
    )
    final_pose[:, :3, :3] = merged["rotation_world"]
    final_pose[:, :3, 3] = merged["translation_world"]
    merged["T_world_r_base_link"] = final_pose
    return merged


def main() -> None:
    args = parse_args()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    global_report = []
    catalog_rows = []
    for case in selection["scenes"]:
        scene = int(case["scene_index"])
        view = int(case["view_index"])
        case_root = (
            OUTPUT_ROOT
            / f"scene_{scene:04d}_view_{view:04d}"
            / "02_predictions"
        )
        raw_path = case_root / "balanced_raw_predictions.npz"
        diagnostics_path = case_root / (
            "balanced_filtered_predictions_all_diagnostics.npz"
        )
        if not raw_path.is_file() or not diagnostics_path.is_file():
            raise FileNotFoundError(
                f"缺少{raw_path}或{diagnostics_path}；"
                "请先运行01_run_network_and_filter.py"
            )
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        with np.load(diagnostics_path, allow_pickle=False) as archive:
            diagnostics = {key: archive[key] for key in archive.files}
        source = merge_raw_and_diagnostics(raw, diagnostics)
        score = np.asarray(source["score"], dtype=np.float64)
        target = np.asarray(source["target_segmentation_id"], dtype=np.int64)
        rotation = np.asarray(source["rotation_world"], dtype=np.float64)
        translation = np.asarray(source["translation_world"], dtype=np.float64)

        selected_indices = []
        selected_pose_ranks = []
        selected_filter_tiers = []
        object_reports = []
        for scene_object in case["objects"]:
            object_id = int(scene_object["segmentation_id"])
            object_mask = target == object_id
            enhanced_count = int(
                np.logical_and(object_mask, source["enhanced_keep_mask"]).sum()
            )
            strict_count = int(
                np.logical_and(object_mask, source["strict_training_keep_mask"]).sum()
            )
            filter_tier = (
                "enhanced_full_hand_and_path"
                if enhanced_count >= 2
                else "official_training_strict_pregrasp_fallback"
            )
            eligible_mask = (
                source["enhanced_keep_mask"]
                if enhanced_count >= 2
                else source["strict_training_keep_mask"]
            )
            available = np.flatnonzero(object_mask & eligible_mask)
            available = available[np.argsort(-score[available])]
            chosen = []
            diversity_relaxed = False
            difference = None
            if len(available):
                chosen.append(int(available[0]))
            if len(available) >= 2:
                first = chosen[0]
                for candidate in available[1:]:
                    candidate = int(candidate)
                    trans_delta = float(
                        np.linalg.norm(translation[candidate] - translation[first])
                    )
                    rot_delta = rotation_distance_deg(
                        rotation[first], rotation[candidate]
                    )
                    if (
                        trans_delta >= args.translation_diversity_m
                        or rot_delta >= args.rotation_diversity_deg
                    ):
                        chosen.append(candidate)
                        difference = (trans_delta, rot_delta)
                        break
                if len(chosen) == 1:
                    chosen.append(int(available[1]))
                    diversity_relaxed = True
                    difference = (
                        float(np.linalg.norm(translation[chosen[1]] - translation[first])),
                        rotation_distance_deg(rotation[first], rotation[chosen[1]]),
                    )

            for pose_rank, index in enumerate(chosen):
                selected_indices.append(index)
                selected_pose_ranks.append(pose_rank)
                selected_filter_tiers.append(filter_tier)
                catalog_rows.append(
                    {
                        "scene_index": scene,
                        "view_index": view,
                        "object_segmentation_id": object_id,
                        "object_code": scene_object["object_code"],
                        "pose_rank": pose_rank,
                        "source_candidate_index": int(
                            source["source_candidate_index"][index]
                        ),
                        "score": float(score[index]),
                        "selection_filter_tier": filter_tier,
                        "final_table_clearance_m": float(
                            source["final_table_clearance_m"][index]
                        ),
                        "final_non_target_clearance_m": float(
                            source["final_non_target_clearance_m"][index]
                        ),
                    }
                )
            object_reports.append(
                {
                    "segmentation_id": object_id,
                    "object_code": scene_object["object_code"],
                    "enhanced_candidate_count": enhanced_count,
                    "strict_training_candidate_count": strict_count,
                    "selection_filter_tier": filter_tier,
                    "eligible_candidate_count": int(len(available)),
                    "selected_count": len(chosen),
                    "selected_filtered_indices": chosen,
                    "source_candidate_indices": [
                        int(source["source_candidate_index"][index]) for index in chosen
                    ],
                    "scores": [float(score[index]) for index in chosen],
                    "second_pose_translation_difference_m": (
                        None if difference is None else difference[0]
                    ),
                    "second_pose_rotation_difference_deg": (
                        None if difference is None else difference[1]
                    ),
                    "diversity_relaxed": diversity_relaxed,
                }
            )

        selected_array = np.asarray(selected_indices, dtype=np.int64)
        selected = candidate_subset(source, selected_array)
        selected["selected_filtered_index"] = selected_array
        selected["pose_rank_within_object"] = np.asarray(
            selected_pose_ranks, dtype=np.int64
        )
        selected["selection_filter_tier"] = np.asarray(selected_filter_tiers)
        output_npz = case_root / "selected_top2.npz"
        with output_npz.open("wb") as stream:
            np.savez_compressed(stream, **selected)
        complete = all(item["selected_count"] == 2 for item in object_reports)
        report = {
            "schema_version": 1,
            "status": (
                "two_candidates_per_object_selected"
                if complete
                else "incomplete_some_objects_have_fewer_than_two_candidates"
            ),
            "scene_index": scene,
            "view_index": view,
            "input_balanced_raw_prediction": raw_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "input_filter_diagnostics": diagnostics_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "output_npz": output_npz.relative_to(PROJECT_ROOT).as_posix(),
            "selected_total": len(selected_indices),
            "translation_diversity_m": args.translation_diversity_m,
            "rotation_diversity_deg": args.rotation_diversity_deg,
            "objects": object_reports,
        }
        output_json = output_npz.with_suffix(".json")
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        global_report.append(report)
        print(
            f"scene={scene:04d} view={view:04d} selected={len(selected_indices)}/12 "
            f"complete={complete}"
        )

    all_complete = all(item["selected_total"] == 12 for item in global_report)
    summary = {
        "schema_version": 1,
        "status": "complete" if all_complete else "incomplete",
        "scene_count": len(global_report),
        "selected_total": sum(item["selected_total"] for item in global_report),
        "expected_total": len(global_report) * 12,
        "scenes": [
            {
                "scene_index": item["scene_index"],
                "view_index": item["view_index"],
                "selected_total": item["selected_total"],
                "status": item["status"],
            }
            for item in global_report
        ],
    }
    path = OUTPUT_ROOT / "selected_top2_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    catalog_path = OUTPUT_ROOT / "selected_pose_catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog_rows[0]))
        writer.writeheader()
        writer.writerows(catalog_rows)
    print(f"summary={path}")
    print(f"catalog={catalog_path}")


if __name__ == "__main__":
    main()
