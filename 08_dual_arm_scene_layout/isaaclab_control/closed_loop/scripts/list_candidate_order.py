#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--limit", type=int, default=128)
    args = p.parse_args()
    with np.load(args.prediction, allow_pickle=False) as a:
        order = np.asarray(a["target_score_descending_candidate_index"], dtype=np.int64)
        score = np.asarray(a["score"], dtype=np.float64)
        graspness = np.asarray(a["graspness"], dtype=np.float64)
        log_prob = np.asarray(a["log_prob"], dtype=np.float64)
    rows = []
    for rank, idx in enumerate(order[: max(0, args.limit)]):
        i = int(idx)
        rows.append({
            "target_rank": rank,
            "candidate_index": i,
            "score": float(score[i]),
            "graspness": float(graspness[i]),
            "log_prob": float(log_prob[i]),
        })
    print(json.dumps(rows, ensure_ascii=False))

if __name__ == "__main__":
    main()
