import torch
import torch.nn as nn


class MLP(nn.Module):
    """Global MLP that predicts normalized body accelerations from a full state."""

    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        state_mean=None,
        state_std=None,
        num_hidden_layers=3,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive.")

        if state_mean is None:
            state_mean = torch.zeros(input_dim)

        if state_std is None:
            state_std = torch.ones(input_dim)

        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std.clamp_min(1e-8))

        layers = []
        current_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Return normalized accelerations with the same leading axes as ``x``."""
        z = (x - self.state_mean) / self.state_std
        return self.net(z)
