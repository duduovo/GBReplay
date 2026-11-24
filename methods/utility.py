def get_graph_class_ratio(graph, class_id):
    """Return the share of training nodes assigned to one class."""
    labels = graph.y[graph.train_mask]
    return (labels == class_id).sum().item() / len(labels)
