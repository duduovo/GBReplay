import torch
import torch.nn.functional as F


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

    def initialize(self):
        for layer in self.layers:
            # torch.nn.init.normal_(layer.lin.weight.data)
            layer.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
        x = self.layers[-1](x, adj_t)
        return x

    def encode(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x, adj_t)
            x = F.relu(x)
        return x

def train_node_classifier(model, data, optimizer, weight=None, n_epoch=300, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    for epoch in range(n_epoch):
        if incremental_cls:
            out = model(data)[:, 0:incremental_cls[1]]
        else:
            out = model(data)

        loss = ce(out[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def train_node_classifier_batch(model, batches, optimizer, n_epoch=300, incremental_cls=None, weight=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    for _ in range(n_epoch):
        for data in batches:
            if incremental_cls:
                out = model(data)[:, 0:incremental_cls[1]]
            else:
                out = model(data)

            loss = ce(out[data.train_mask], data.y[data.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def eval_node_classifier(model, data, incremental_cls=None, test_mask=None, labels=None):
    model.eval()
    mask = data.test_mask if test_mask is None else test_mask
    target = data.y if labels is None else labels
    pred = model(data)[mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
    correct = (pred == target[mask] - incremental_cls[0]).sum()
    acc = int(correct) / int(mask.sum())
    return acc
