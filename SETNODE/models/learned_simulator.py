import sys
from pathlib import Path

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.rollout.integrators import velocity_verlet_step
from SETNODE.graph_builder import build_graph


class LearnedSimulator(nn.Module):
    """
    Wraps the graph network with force unnormalization
    and one-step physics integration.
    """

    def __init__(
        self,
        graph_network,
        force_std,
        dt,
        edge_feature_dim=None,
        mass_feature_mode="raw",
    ):
        super().__init__()

        self.graph_network = graph_network
        self.dt = dt
        self.edge_feature_dim = edge_feature_dim
        self.mass_feature_mode = mass_feature_mode

        # Register as buffers so they move with model.to(device)
        # force_std has shape [1,1,1,1] we change it to [1] for it to broadcast from [num_nodes, spatial_dim] properly
        # -1 in reshape tells pytorch to use total number of elemts as dimension, i.e [2,3].reshape(-1) has shape [6]
        self.register_buffer("force_std", force_std.reshape(-1))

    def predict_acceleration(self, positions, masses):
        graph = build_graph(
            positions=positions,
            masses=masses,
            mass_feature_mode=self.mass_feature_mode,
        )
        if self.edge_feature_dim is not None:
            actual_edge_feature_dim = graph["edge_features"].shape[-1]
            if actual_edge_feature_dim != self.edge_feature_dim:
                raise ValueError(
                    "The graph builder produces "
                    f"{actual_edge_feature_dim} edge features, but the checkpoint "
                    f"expects {self.edge_feature_dim}. Retrain the model with the "
                    "current coordinate-independent edge features."
                )

        # The one-step loss trains the graph network to predict normalized force.
        predicted_force_normalized = self.graph_network(graph)

        predicted_force = predicted_force_normalized * self.force_std
        return predicted_force / masses

    def forward(self, positions, velocities, masses):

        initial_acceleration = self.predict_acceleration(positions, masses)

        next_positions, next_velocities, _ = velocity_verlet_step(
            positions,
            velocities,
            self.predict_acceleration,
            self.dt,
            masses,
            acceleration=initial_acceleration,
        )

        return next_positions, next_velocities, initial_acceleration
