import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, *args):
        """Accept either ``model(x, edge_index)`` or ``model(data)``."""
        if len(args) == 1 and isinstance(args[0], Data):
            data = args[0]
            x = data.x
            edge_index = self._get_edge_index(data)
        elif len(args) == 2:
            x, edge_index = args
        else:
            raise ValueError("Invalid arguments. Expected Data object or (x, edge_index)")

        x = x.to(next(self.parameters()).device)
        edge_index = edge_index.to(x.device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def _get_edge_index(self, data):
        if hasattr(data, 'adj_t'):
            row, col, _ = data.adj_t.coo()
            return torch.stack([row, col], dim=0)
        return data.edge_index
