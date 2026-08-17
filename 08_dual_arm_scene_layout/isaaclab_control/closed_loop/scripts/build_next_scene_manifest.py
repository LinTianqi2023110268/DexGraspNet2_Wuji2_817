#!/usr/bin/env python3
"""Persist the final simulated object poses as the next closed-loop scene state."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def quat_wxyz_to_R(q):
    w,x,y,z = np.asarray(q,dtype=np.float64)
    n = np.linalg.norm([w,x,y,z])
    if n <= 1e-12: raise ValueError("zero quaternion")
    w,x,y,z = np.asarray([w,x,y,z])/n
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--settled-manifest",type=Path,required=True)
    p.add_argument("--physical-replay",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads(a.settled_manifest.read_text(encoding="utf-8"))
    world_from_source=np.asarray(manifest["world_from_source_zone"],dtype=np.float64)
    source_from_world=np.linalg.inv(world_from_source)
    with np.load(a.physical_replay,allow_pickle=False) as z:
        poses=np.asarray(z["object_pose_world_wxyz"][-1],dtype=np.float64)
        metadata=json.loads(str(np.asarray(z["metadata_json"]).item()))
    meta_objects=metadata["objects"]
    if len(poses)!=len(meta_objects):
        raise RuntimeError("replay object count mismatch")
    by_seg={}
    for pose,meta in zip(poses,meta_objects):
        seg=int(meta["segmentation_id"])
        T=np.eye(4)
        T[:3,3]=pose[:3]
        T[:3,:3]=quat_wxyz_to_R(pose[3:7])
        by_seg[seg]=T
    new_objects=[]
    for record in manifest["objects"]:
        seg=int(record["segmentation_id"])
        if seg not in by_seg:
            raise RuntimeError(f"missing replay pose for segmentation {seg}")
        Tw=by_seg[seg]
        Ts=source_from_world@Tw
        rr=dict(record)
        rr["T_world_centered_object"]=Ts.tolist()
        rr["pose_world_object"]=Ts.tolist()
        rr["settled_pose_layout_world"]=Tw.tolist()
        new_objects.append(rr)
    out=dict(manifest)
    out["schema_version"]=3
    out["status"]="closed_loop_state_from_previous_physical_cycle"
    out["objects"]=new_objects
    out["previous_physical_replay"]=str(a.physical_replay.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","output":str(a.output.resolve()),"objects":len(new_objects)},ensure_ascii=False))

if __name__=="__main__":
    main()
