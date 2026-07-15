import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.rollout.integrators import rk4_step
from graph_builder import build_graph


class LearnedSimulator(nn.Module):
    """
    Wraps the graph network with acceleration unnormalization
    and one-step physics integration.
    """

    def __init__(
        self, graph_network, acc_mean, acc_std, dt=0.01, edge_feature_dim=None
    ):
        super().__init__()

        self.graph_network = graph_network
        self.dt = dt
        self.edge_feature_dim = edge_feature_dim

        # Keep acc_mean as a buffer for checkpoint compatibility, but GNS uses
        # the EGNS zero-mean convention when unnormalizing accelerations.
        self.register_buffer("acc_mean", acc_mean)
        self.register_buffer("acc_std", acc_std)

    def predict_acceleration(self, positions, velocities, masses):
        graph = build_graph(
            positions=positions,
            velocities=velocities,
            masses=masses,
        )
        if self.edge_feature_dim is not None:
            graph["edge_features"] = graph["edge_features"][:, : self.edge_feature_dim]

        predicted_acceleration_normalized = self.graph_network(graph)

        predicted_acceleration = predicted_acceleration_normalized * self.acc_std.view(
            -1
        )

        return predicted_acceleration

    def forward(self, positions, velocities, masses):
        dt = self.dt
        dim = positions.shape[-1]

        initial_acceleration = self.predict_acceleration(positions, velocities, masses)
        state = torch.cat([positions, velocities], dim=-1)

        def dynamics(current_state):
            current_positions = current_state[..., :dim]
            current_velocities = current_state[..., dim:]
            current_acceleration = self.predict_acceleration(
                current_positions,
                current_velocities,
                masses,
            )
            return torch.cat([current_velocities, current_acceleration], dim=-1)

        next_state = rk4_step(state, dynamics, dt)

        next_positions = next_state[..., :dim]
        next_velocities = next_state[..., dim:]

        return next_positions, next_velocities, initial_acceleration
