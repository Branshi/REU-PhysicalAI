import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, node_dim, edge_dim, attention_dim):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim
        self.W = nn.Linear(input_dim, attention_dim)
        self.a = nn.Linear(attention_dim, 1, bias=False)
        self.V = nn.Linear(edge_dim, attention_dim)

    # hi : [N - 1, node_dim], h_i repeated N - 1 times
    # hj : [N - 1, node_dim]
    def forward(self, h_i, h_j, e_ij):
        x_ij = torch.cat([h_i, h_j, e_ij], dim=-1)
        hidden = F.leaky_relu(self.W(x_ij), negative_slope=0.2)
        score = self.a(hidden).squeeze(-1)
        # alpha : [N - 1]
        alpha = torch.softmax(score, dim=0)
        # e_ij = [N - 1, feature_dim], these are the edges from fixed body i to all other bodies excluding i.
        # values [N - 1, attention_dim]
        values = self.V(e_ij)
        # head : [attention_dim], attention vector for fixed node i, will be later stacked to form a matrix.
        # centered residual attention
        inverse_neighbors = 1 / h_j.shape[0]
        head = torch.sum((alpha - inverse_neighbors).unsqueeze(-1) * values, dim=0)
        #
        return head


class MultiHeadAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads):
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError(
                "Number of attention heads does not divide node representation dimension."
            )
        self.W_o = nn.Linear(node_dim, node_dim)
        self.num_heads = num_heads
        self.attention_dim = node_dim // num_heads
        self.heads = nn.ModuleList([
            Attention(node_dim, edge_dim, self.attention_dim) for _ in range(num_heads)
        ])

    def forward(self, h_i, h_j, e_ij):
        head_outputs = [head(h_i, h_j, e_ij) for head in self.heads]
        M = torch.cat(head_outputs, dim=-1)
        attention = self.W_o(M)
        return attention
