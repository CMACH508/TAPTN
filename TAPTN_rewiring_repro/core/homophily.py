"""Node-averaged edge homophily (paper Eq. 1 / supplementary `homophily.ipynb`)."""
from __future__ import annotations

import numpy as np


def homophily_from_pkl_data(data, labels):
    """Match the supplementary notebook, not a directed skip-isolated mean.

    Each stored directed edge is treated as undirected (both endpoints get the
    neighbour). Isolated nodes add 0 to the sum but remain in the denominator
    ``num_nodes``. Duplicate neighbour entries (from already-bidirectional
    edges) do not change the per-node ratio.
    """
    ei = data.edge_index
    if hasattr(ei, "cpu"):
        ei = ei.cpu()
    if hasattr(ei, "t"):
        edges = ei.t().tolist()
    else:
        src, dst = np.asarray(ei[0]), np.asarray(ei[1])
        edges = list(zip(src.tolist(), dst.tolist()))
    labels = np.asarray(labels)
    n = int(len(labels))
    adj = [[] for _ in range(n)]
    for i, j in edges:
        i, j = int(i), int(j)
        adj[i].append(j)
        adj[j].append(i)
    hsum = 0.0
    for i in range(n):
        neigh = adj[i]
        if len(neigh) == 0:
            continue
        same = sum(1 for j in neigh if labels[i] == labels[j])
        hsum += same / len(neigh)
    return float(hsum / n) if n else 0.0
