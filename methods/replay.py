import torch
from progressbar import progressbar

from methods.utility import get_graph_class_ratio


class Replay:
    """Base class for methods that build one compact replay graph per task."""

    def __init__(self, model, tasks, budget, m_update, device):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.budgets = self._assign_budget_per_class(budget)
        self.device = device
        self.memory_bank = []
        self.strategy = m_update

    def observer(self):
        """Build and store one replay graph for every task."""
        for task_id in progressbar(range(len(self.tasks)), redirect_stdout=True):
            task = self.tasks[task_id]
            replayed_graph = self.memorize(task, self.budgets[task_id])
            self.memory_bank.append(replayed_graph)
            task.to("cpu")
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        return self.memory_bank

    def memorize(self, task, budgets):
        """Select replay data for one task using per-class node budgets."""
        raise NotImplementedError("Subclasses must implement memorize().")

    def _assign_budget_per_class(self, budget):
        """Allocate each task's replay budget across its classes."""
        all_budgets = []
        for task in self.tasks:
            classes = torch.as_tensor(
                list(getattr(task, "classes", torch.unique(task.y).tolist())),
                dtype=torch.long,
            )
            task_budgets = []
            for class_id in classes:
                class_ratio = get_graph_class_ratio(task, class_id)
                class_budget = max(1, int(budget * class_ratio))
                task_budgets.append(class_budget)

            gap = int(budget) - sum(task_budgets)
            for index in range(max(0, gap)):
                task_budgets[index % len(classes)] += 1
            all_budgets.append(task_budgets)
        return all_budgets
