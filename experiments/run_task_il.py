"""Reproduce the GBReplay rows in the Task-IL main-results table."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"
DATASET_BUDGETS = {
    "corafull": 2,
    "arxiv": 26,
    "reddit": 34,
    "products": 319,
}
GBREPLAY_ARGS = {
    "a": 0.2,
    "n_encoders": 100,
    "feat_lr": 0.001,
    "hid_dim": 512,
    "emb_dim": 256,
    "n_layers": 2,
    "hop": 1,
    "activation": True,
    "class_factor": 0.3,
    "ball_score_w_sz": 0.4,
    "ball_score_w_ig": 0.4,
    "ball_score_w_p": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GBReplay Task-IL main-table experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_BUDGETS),
        default=list(DATASET_BUDGETS),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root or (
        ROOT / "results" / f"task_il_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        run_dir = output_root / dataset
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        command = [
            sys.executable,
            str(TRAIN),
            "--dataset-name",
            dataset,
            "--cls-per-task",
            "2",
            "--cls-epoch",
            str(args.epochs),
            "--budget",
            str(DATASET_BUDGETS[dataset]),
            "--m-update",
            "all",
            "--IL",
            "taskIL",
            "--device",
            args.device,
            "--repeat",
            str(args.repeat),
            "--seed",
            str(args.seed),
            "--result-path",
            str(run_dir),
            "--GBR-args",
            repr(GBREPLAY_ARGS),
        ]
        print(f"Running Task-IL on {dataset}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        rows.append(
            {
                "dataset": dataset,
                "budget": DATASET_BUDGETS[dataset],
                "repeat": args.repeat,
                "exit_code": process.returncode,
                "log": str(log_path),
            }
        )
        status = "completed" if process.returncode == 0 else "failed"
        print(f"{dataset}: {status} (exit code {process.returncode})", flush=True)

    summary_path = output_root / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=("dataset", "budget", "repeat", "exit_code", "log"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")
    return 1 if any(row["exit_code"] != 0 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
