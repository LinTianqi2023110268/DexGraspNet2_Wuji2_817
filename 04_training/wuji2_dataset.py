"""Multi-scene Wuji2 dataset with the DexGraspNet2 loader tensor contract."""

from __future__ import annotations

import collections.abc as container_abcs
import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import Dataset

from wuji2_dgn2.adapter_common import load_manifest_ancestor_with_key
from wuji2_dgn2.project import project_path


class Wuji2SceneDataset(Dataset):
    """Read isolated Stage-03/04 files and emit official training keys.

    The grasp sampling mirrors ``DexGraspNet2/src/utils/dataset.py``:
    sample 128 candidates across non-empty object grasp files, randomize them,
    retain candidates whose complete-surface reference point is within 6 mm of
    the current single-view cloud, then sample 64 matched labels with
    replacement.
    """

    def __init__(
        self,
        adapter_output_root: str | Path,
        scene_indices=(0,),
        is_train: bool = True,
        repeat_length: int = 100000,
        sample_total: int = 128,
        k: int = 64,
        max_point_dis: float = 0.006,
        voxel_size: float = 0.005,
    ):
        self.root = Path(adapter_output_root).resolve()
        self.scene_indices = tuple(int(value) for value in scene_indices)
        self.is_train = bool(is_train)
        self.repeat_length = int(repeat_length)
        self.sample_total = int(sample_total)
        self.k = int(k)
        self.max_point_dis = float(max_point_dis)
        self.voxel_size = float(voxel_size)
        if self.sample_total < self.k:
            raise ValueError("sample_total must be >= k")
        self.samples = []
        self.excluded_empty_views = []
        self.grasps = {}
        self.joint_order = None
        for scene_index in self.scene_indices:
            stage04_root = (
                self.root
                / "grasp_label_stages"
                / "04_single_view_training_labels"
                / f"scene_{scene_index:04d}"
            )
            stage04 = json.loads(
                (stage04_root / "stage_manifest.json").read_text(encoding="utf-8")
            )
            if not stage04.get("training_ready"):
                raise RuntimeError(f"Stage 04 is not training-ready: {stage04_root}")
            stage03_path = project_path(stage04["input_stage_manifest"])
            stage03 = json.loads(stage03_path.read_text(encoding="utf-8"))
            stage01 = load_manifest_ancestor_with_key(stage03, "label_contract")
            joint_order = tuple(stage01["label_contract"]["joint_order"])
            if self.joint_order is None:
                self.joint_order = joint_order
            elif self.joint_order != joint_order:
                raise RuntimeError("Joint order differs between scenes")
            scene_grasps = {}
            for record in stage03["object_records"]:
                if int(record["grasp_count"]) == 0:
                    continue
                with np.load(project_path(record["grasp_npz"])) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                if arrays["qpos"].shape[1] != len(joint_order):
                    raise RuntimeError("Wuji2 qpos/joint-order dimension mismatch")
                scene_grasps[int(record["segmentation_id"])] = arrays
            if not scene_grasps:
                raise RuntimeError(f"No non-empty grasp files in scene {scene_index}")
            self.grasps[scene_index] = scene_grasps
            for view_record in stage04["view_records"]:
                if int(view_record.get("total_available_grasp_count", 0)) <= 0:
                    self.excluded_empty_views.append(
                        (scene_index, int(view_record["view_index"]))
                    )
                    continue
                self.samples.append(
                    (
                        scene_index,
                        int(view_record["view_index"]),
                        project_path(view_record["output_npz"]),
                    )
                )
        if self.joint_order is None or len(self.joint_order) != 20:
            raise RuntimeError("Expected exactly 20 Wuji2 joints")

    def __len__(self):
        return self.repeat_length if self.is_train else len(self.samples)

    @staticmethod
    def _camera_from_world(world_from_camera: np.ndarray) -> np.ndarray:
        return np.linalg.inv(world_from_camera).astype(np.float32)

    @staticmethod
    def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        return (
            points @ transform[:3, :3].T + transform[:3, 3]
        ).astype(np.float32)

    def _sample_grasps(self, scene_index: int, view: dict) -> dict[str, np.ndarray]:
        object_grasps = self.grasps[scene_index]
        object_ids = sorted(object_grasps)
        object_draws = np.random.randint(0, len(object_ids), self.sample_total)
        candidates = []
        for object_slot, object_id in enumerate(object_ids):
            number = int((object_draws == object_slot).sum())
            if not number:
                continue
            arrays = object_grasps[object_id]
            indices = np.random.choice(len(arrays["point"]), number, replace=True)
            for index in indices:
                candidates.append((object_id, int(index)))
        candidates = [candidates[index] for index in np.random.permutation(len(candidates))]
        cloud = view["point_clouds"]
        world_from_camera = view["T_world_camera"]
        camera_from_world = self._camera_from_world(world_from_camera)
        available = []
        centers = []
        camera_rotations = []
        camera_translations = []
        qposes = []
        for object_id, grasp_index in candidates:
            arrays = object_grasps[object_id]
            point_camera = self._transform_points(
                camera_from_world, arrays["point"][[grasp_index]]
            )[0]
            distance = np.linalg.norm(cloud - point_camera, axis=1)
            nearest_index = int(distance.argmin())
            if float(distance[nearest_index]) > self.max_point_dis:
                continue
            world_rotation = arrays["rotation"][grasp_index]
            world_translation = arrays["translation"][grasp_index]
            camera_rotations.append(camera_from_world[:3, :3] @ world_rotation)
            camera_translations.append(
                camera_from_world[:3, :3] @ world_translation
                + camera_from_world[:3, 3]
            )
            qposes.append(arrays["qpos"][grasp_index])
            centers.append(nearest_index)
            available.append(len(available))
            # Official dataset.py stops scanning the 128 shuffled candidates
            # as soon as K visible labels have been found, then resamples K
            # entries with replacement from that prefix.
            if len(available) >= self.k:
                break
        if not available:
            raise RuntimeError(
                f"No grasp reference point is within {self.max_point_dis} m of view"
            )
        selected = np.random.choice(len(available), self.k, replace=True)
        return {
            "rot": np.asarray(camera_rotations, np.float32)[selected],
            "trans": np.asarray(camera_translations, np.float32)[selected],
            "qpos": np.asarray(qposes, np.float32)[selected],
            "centers": np.asarray(centers, np.float32)[selected],
            "matched_before_k_resample": np.asarray([len(available)], np.int64),
        }

    @staticmethod
    def _augment(ret: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        theta = np.random.rand() * 2.0 * np.pi
        rotation = np.asarray(
            [
                [np.cos(theta), np.sin(theta), 0.0],
                [-np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        ret["point_clouds"] = np.einsum(
            "ij,nj->ni", rotation, ret["point_clouds"]
        )
        ret["trans"] = np.einsum("ij,nj->ni", rotation, ret["trans"])
        ret["rot"] = np.einsum("ij,njk->nik", rotation, ret["rot"])
        ret["coors"] = ret["point_clouds"] / ret.pop("_voxel_size")
        return ret

    def __getitem__(self, dataset_index: int):
        # The official loader resamples a different random training item when
        # none of its 128 randomly drawn labels is visible in the current
        # camera.  Keep that behavior, but bound it so a broken dataset cannot
        # recurse forever.  Evaluation never substitutes another view.
        maximum_attempts = 32 if self.is_train else 1
        last_error = None
        for attempt in range(maximum_attempts):
            if self.is_train:
                scene_index, view_index, path = self.samples[
                    np.random.randint(0, len(self.samples))
                ]
            else:
                scene_index, view_index, path = self.samples[dataset_index]
            with np.load(path) as archive:
                view = {key: archive[key] for key in archive.files}
            try:
                labels = self._sample_grasps(scene_index, view)
                break
            except RuntimeError as exc:
                last_error = exc
                if not self.is_train:
                    raise
        else:
            raise RuntimeError(
                f"No usable randomly sampled labels after {maximum_attempts} "
                "official-style training resamples"
            ) from last_error
        ret = {
            "scene": np.asarray([scene_index], dtype=np.int64),
            "view": np.asarray([view_index], dtype=np.int64),
            "point_clouds": view["point_clouds"].astype(np.float32),
            "coors": view["coors"].astype(np.float32),
            "feats": view["feats"].astype(np.float32),
            "seg": view["seg"].astype(np.int64),
            "objectness": view["objectness"].astype(np.int64),
            "graspness": view["graspness_log_target"].astype(np.float32),
            "has_graspness": np.asarray([1], dtype=np.int64),
            "_voxel_size": self.voxel_size,
            **labels,
        }
        if self.is_train:
            ret = self._augment(ret)
        else:
            ret.pop("_voxel_size")
        return ret


def minkowski_collate_fn(list_data):
    """Copy of the official sparse-collate tensor contract."""
    coordinates_batch, features_batch = ME.utils.sparse_collate(
        [item["coors"] for item in list_data],
        [item["feats"] for item in list_data],
    )
    coordinates_batch, features_batch, original2quantize, quantize2original = (
        ME.utils.sparse_quantize(
            coordinates_batch,
            features_batch,
            return_index=True,
            return_inverse=True,
        )
    )
    result = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "original2quantize": original2quantize,
        "quantize2original": quantize2original,
    }

    def collate(batch):
        if type(batch[0]).__module__ == "numpy":
            return torch.stack([torch.from_numpy(value) for value in batch], 0)
        if isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in value] for value in batch]
        if isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key in ("coors", "feats"):
                    continue
                result[key] = collate([value[key] for value in batch])
            return result
        raise TypeError(f"Unsupported collate type: {type(batch[0])}")

    return collate(list_data)
