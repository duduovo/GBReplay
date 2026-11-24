import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Batch

from backbones.gnn import (
    eval_node_classifier,
    train_node_classifier,
    train_node_classifier_batch,
)
from data_stream import Streaming
from utilities import (
    get_backbone_model,
    get_cgl_model,
    get_dataset,
    get_result_file_name,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "data"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results"
DEFAULT_GBR_ARGS = {
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


def format_mean_std(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if values.size > 1 else 0.0
    return f"{mean:.1f}±{std:.1f}"


def load_or_create_stream(args, dataset):
    stream_dir = args.data_dir / "streaming"
    stream_dir.mkdir(parents=True, exist_ok=True)
    stream_path = stream_dir / (
        f"{args.dataset_name}_cpt{args.cls_per_task}_seed{args.seed}.streaming"
    )
    if stream_path.exists():
        return torch.load(stream_path)

    data_stream = Streaming(args.cls_per_task, dataset)
    torch.save(data_stream, stream_path)
    return data_stream


def build_memory_banks(args, dataset, data_stream):
    memory_dir = args.result_path / "memory_bank"
    memory_dir.mkdir(parents=True, exist_ok=True)
    result_name = get_result_file_name(args)
    memory_banks = []

    for repeat_index in range(args.repeat):
        memory_path = memory_dir / f"{result_name}_repeat_{repeat_index}.pt"
        if memory_path.exists() and not args.rebuild_memory:
            memory_bank = torch.load(memory_path)
        else:
            model = get_backbone_model(dataset, data_stream, args)
            gbreplay = get_cgl_model(model, data_stream, args)
            memory_bank = gbreplay.observer()
            torch.save(memory_bank, memory_path)
        memory_banks.append(memory_bank)

    return memory_banks


def train_on_replay(model, optimizer, memory_bank, task_index, args):
    current_replay = memory_bank[task_index]
    max_class = int(torch.unique(current_replay.y)[-1])

    if args.dataset_name == "products":
        batches = memory_bank[: task_index + 1]
        for batch in batches:
            batch.to(args.device)
        train_node_classifier_batch(
            model,
            batches,
            optimizer,
            n_epoch=args.cls_epoch,
            incremental_cls=(0, max_class + 1),
        )
        for batch in batches:
            batch.to("cpu")
    else:
        replay_graph = Batch.from_data_list(memory_bank[: task_index + 1])
        replay_graph.to(args.device, "x", "y", "adj_t")
        train_node_classifier(
            model,
            replay_graph,
            optimizer,
            n_epoch=args.cls_epoch,
            incremental_cls=(0, max_class + 1),
        )
        replay_graph.to("cpu")

    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return max_class


def evaluate(args, dataset, data_stream, memory_banks):
    final_aps = []
    mean_aps = []
    final_afs = []
    performance_matrices = []

    for memory_bank in memory_banks:
        model = get_backbone_model(dataset, data_stream, args)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.005, weight_decay=5e-4
        )
        tasks = data_stream.tasks
        matrix = torch.zeros(len(tasks), len(tasks))
        ap_history = []

        for task_index in range(len(memory_bank)):
            max_class = train_on_replay(
                model, optimizer, memory_bank, task_index, args
            )
            accuracies = []

            for eval_index in range(task_index + 1):
                task = tasks[eval_index].to(
                    args.device, "x", "y", "adj_t"
                )
                if args.IL == "classIL":
                    class_range = (0, max_class + 1)
                else:
                    task_max_class = int(torch.unique(task.y)[-1])
                    class_range = (
                        task_max_class + 1 - data_stream.cls_per_task,
                        task_max_class + 1,
                    )
                accuracy = (
                    eval_node_classifier(
                        model, task, incremental_cls=class_range
                    )
                    * 100
                )
                accuracies.append(accuracy)
                matrix[task_index, eval_index] = accuracy
                task.to("cpu")
                print(
                    f"T{eval_index} {accuracy:.2f}",
                    end="|",
                    flush=True,
                )

            average_precision = sum(accuracies) / len(accuracies)
            ap_history.append(average_precision)
            forgetting = 0.0
            if task_index > 0:
                forgetting = sum(
                    matrix[task_index, prior] - matrix[prior, prior]
                    for prior in range(task_index)
                ) / task_index
            print(
                f"AP: {average_precision:.2f}, AF: {forgetting:.2f}",
                flush=True,
            )

        final_aps.append(ap_history[-1])
        mean_aps.append(float(np.mean(ap_history)))
        final_afs.append(float(forgetting))
        performance_matrices.append(matrix)

    print(f"AP: {format_mean_std(final_aps)}")
    print(f"mAP: {format_mean_std(mean_aps)}")
    print(f"AF: {format_mean_std(final_afs)}")
    return performance_matrices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate GBReplay for graph continual learning."
    )
    parser.add_argument(
        "--dataset-name",
        choices=("corafull", "arxiv", "reddit", "products"),
        default="corafull",
    )
    parser.add_argument("--cls-per-task", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--cls-epoch", type=int, default=300)
    parser.add_argument(
        "--budget",
        type=int,
        default=2,
        help="Total replay budget per task, distributed across its classes.",
    )
    parser.add_argument("--m-update", default="all")
    parser.add_argument("--GBR-args", default=repr(DEFAULT_GBR_ARGS))
    parser.add_argument("--IL", choices=("classIL", "taskIL"), default="classIL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--rebuild-memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    seed_everything(args.seed)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.result_path.mkdir(parents=True, exist_ok=True)
    performance_dir = args.result_path / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(args)
    data_stream = load_or_create_stream(args, dataset)
    memory_banks = build_memory_banks(args, dataset, data_stream)
    matrices = evaluate(args, dataset, data_stream, memory_banks)
    torch.save(
        matrices,
        performance_dir / f"{get_result_file_name(args)}.pt",
    )


if __name__ == "__main__":
    main()
