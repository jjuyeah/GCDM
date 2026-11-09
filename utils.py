import torch
from torch_geometric.utils import to_undirected


def to_undirected_src(A, topk=3):
    row = torch.arange(A.shape[0]).unsqueeze(1).repeat(1, topk).unsqueeze(2).to(A.device)
    col = A.topk(topk)[1].unsqueeze(2)
    edge_index = torch.cat([row, col], dim=2).reshape(-1, 2).T
    edge_index = to_undirected(edge_index)
    return edge_index


def to_directed_src(A, topk=3):
    row = torch.arange(A.shape[0]).unsqueeze(1).repeat(1, topk).unsqueeze(2).to(A.device)
    col = A.topk(topk)[1].unsqueeze(2)
    edge_index = torch.cat([row, col], dim=2).reshape(-1, 2).T
    return edge_index


def to_directed_tgt(A, topk=3):
    row = A.topk(topk, dim=1)[1].view(-1)
    col = torch.arange(A.shape[0], device=A.device).repeat_interleave(topk)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def to_undirected_tgt(A, topk=3):
    row = A.topk(topk, dim=1)[1].view(-1)
    col = torch.arange(A.shape[0], device=A.device).repeat_interleave(topk)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = to_undirected(edge_index)
    return edge_index


def to_directed_adj(A, k):
    A_flat = A.flatten()
    topk_values, topk_indices = A_flat.topk(k)
    N = A.shape[0]
    row = topk_indices // N
    col = topk_indices % N
    edge_index = torch.stack([row, col], dim=0)
    return edge_index
