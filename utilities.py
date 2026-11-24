import ast

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CoraFull, Reddit

from backbones.gcn import GCN
from methods.GBR import GBR


def get_result_file_name(args):
    """Return a stable identifier for one GBReplay configuration."""
    return (
        f"{args.dataset_name}_budget{args.budget}_GBR_"
        f"{args.IL}_seed{args.seed}"
    )


def get_dataset(args):
    """Load one of the four datasets evaluated in the paper."""
    if args.dataset_name == "corafull":
        return CoraFull(args.data_dir)
    if args.dataset_name == "arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=args.data_dir)
        dataset.data.y.squeeze_()
        return dataset
    if args.dataset_name == "reddit":
        return Reddit(args.data_dir)
    if args.dataset_name == "products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root=args.data_dir)
        dataset.data.y.squeeze_()
        return dataset
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


def get_backbone_model(dataset, data_stream, args):
    """Create the GCN backbone used by the paper experiments."""
    hidden_dim = 256 if args.dataset_name in {"corafull", "arxiv"} else 64
    output_dim = data_stream.n_tasks * args.cls_per_task
    return GCN(dataset.num_features, hidden_dim, output_dim).to(args.device)


def get_cgl_model(model, data_stream, args):
    """Create GBReplay with the command-line configuration."""
    gbr_args = ast.literal_eval(args.GBR_args)
    return GBR(
        model,
        data_stream.tasks,
        args.budget,
        args.m_update,
        args.device,
        gbr_args,
    )
