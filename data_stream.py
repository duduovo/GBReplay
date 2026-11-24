from progressbar import progressbar
from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor


class Streaming:
    """Split a graph dataset into tasks containing consecutive classes."""

    def __init__(self, cls_per_task, dataset):
        self.cls_per_task = cls_per_task
        self.tasks = self.prepare_tasks(dataset)
        self.n_tasks = len(self.tasks)

    def prepare_tasks(self, dataset):
        graph = dataset[0]
        tasks = []
        n_tasks = int(dataset.num_classes / self.cls_per_task)

        for task_id in progressbar(range(n_tasks), redirect_stdout=True):
            start_class = task_id * self.cls_per_task
            classes = list(range(start_class, start_class + self.cls_per_task))
            print(
                f"Task {task_id} labels range: {min(classes)} to {max(classes)}"
            )
            subset = sum(graph.y == cls for cls in classes).nonzero(
                as_tuple=False
            ).squeeze()
            subgraph = graph.subgraph(subset)
            subgraph = RandomNodeSplit(num_val=0.2, num_test=0.2)(subgraph)

            edge_index = subgraph.edge_index
            adjacency = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes),
            )
            subgraph.adj_t = adjacency.t().to_symmetric()
            subgraph.task_id = task_id
            subgraph.classes = classes
            tasks.append(subgraph)

        return tasks
