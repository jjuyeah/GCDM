import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Pooler(nn.Module):
    def __init__(self, num_nodes: int, c_nodes: int):
        super(Pooler, self).__init__()
        self.matrix = nn.Parameter(torch.randn(num_nodes, c_nodes))

    def _get_matrix(self):
        M = F.softmax(self.matrix, dim=1)
        D_c_inv = (1 / M.T.sum(dim=1)).diag()
        return M, D_c_inv
    
    def pooling(self, x, A):
        M, D_c_inv = self._get_matrix()
        x_c = D_c_inv @ M.T @ x
        A_c = M.T @ A @ M
        return x_c, A_c
    
    def unpooling(self, x_c, A_c):
        M, D_c_inv = self._get_matrix()
        x_r = M @ x_c
        A_c_norm = D_c_inv @ A_c @ D_c_inv
        A_r = M @ A_c_norm @ M.T
        return x_r, A_r
    
    def entropy(self):
        M, _ = self._get_matrix()
        entropy = -(M * (M + 1e-8).log()).sum(dim=1)
        return entropy.mean()


class DeepPooler(nn.Module):
    def __init__(self, num_nodes: int, c_nodes_list: List[int]):
        super(DeepPooler, self).__init__()
        self.poolers = nn.ModuleList([Pooler(num_nodes, c_nodes_list[0])])
        for i in range(len(c_nodes_list) - 1):
            self.poolers.append(Pooler(c_nodes_list[i], c_nodes_list[i + 1]))
    
    def pooling(self, x, A):
        for pooler in self.poolers:
            x, A = pooler.pooling(x, A)
        return x, A
    
    def unpooling(self, x, A):
        for pooler in reversed(self.poolers):
            x, A = pooler.unpooling(x, A)
        return x, A
    
    def entropy(self):
        entropy = []
        for pooler in self.poolers:
            entropy.append(pooler.entropy())
        entropy = torch.tensor(entropy)
        return entropy.mean()
