"""Law-agnostic SETNODE ablation with a more expressive potential readout.

This module intentionally avoids embedding a particular force law. It changes
the baseline architecture in four ways:

1. Directed pair energies are scored nonlinearly before they are averaged into
   an exchange-symmetric unordered-pair energy.
2. Distances use a learned multiscale radial-basis representation.
3. Normalized input node and edge features skip directly to the potential
   readout alongside the processed latent features.
4. Node, pair, and global invariant potential branches have learned gates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .attention_head import MultiHeadAttention
from .graph_network import MLP


class LearnableRadialBasis(nn.Module):
    """Embed scalar distances with learned-scale Gaussian radial bases."""

    def __init__(self, num_basis: int, initial_scale: float = 1.0):
        super().__init__()
        if num_basis < 2:
            raise ValueError("num_basis must be at least 2")
        if initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive")

        self.register_buffer("centers", torch.linspace(0.0, 1.0, num_basis))
        initial_width = 1.0 / (num_basis - 1)
        self.log_width = nn.Parameter(torch.tensor(math.log(initial_width)))
        self.log_distance_scale = nn.Parameter(
            torch.tensor(math.log(initial_scale))
        )

    def forward(self, distance_squared, epsilon=0.0):
        softened_distance = torch.sqrt(
            (distance_squared + float(epsilon) ** 2).clamp_min(1e-12)
        )
        distance_scale = self.log_distance_scale.exp().clamp_min(1e-8)
        compact_distance = softened_distance / (
            softened_distance + distance_scale
        )
        width = self.log_width.exp().clamp_min(1e-4)
        standardized_distance = (
            compact_distance - self.centers
        ) / width
        return torch.exp(-0.5 * standardized_distance.square())


class GenericInteractionNetwork(nn.Module):
    """Message-passing block using only learned invariant radial features."""

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        distance_dim=16,
        ffn_dim=256,
        num_hidden_layers=2,
        num_heads=4,
        rbf_initial_scale=1.0,
    ):
        super().__init__()

        self.distance_embedding = LearnableRadialBasis(
            num_basis=distance_dim,
            initial_scale=rbf_initial_scale,
        )
        self.edge_mlp = MLP(
            input_dim=latent_dim * 3 + distance_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )
        self.node_mlp = MLP(
            input_dim=latent_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.multi_attention = MultiHeadAttention(
            latent_dim,
            latent_dim,
            num_heads,
        )
        self.attention_mlp = MLP(
            input_dim=latent_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )
        self.attention_scale = nn.Parameter(
            torch.full((latent_dim,), 1e-2)
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, latent_dim),
        )

    def forward(
        self,
        node_latents,
        edge_latents,
        coordinates,
        senders,
        receivers,
        epsilon,
    ):
        norm_node_latents = self.norm1(node_latents)
        receiver_node_latents = norm_node_latents[receivers]
        sender_node_latents = norm_node_latents[senders]

        relative_position = coordinates[senders] - coordinates[receivers]
        distance_squared = relative_position.square().sum(
            dim=-1,
            keepdim=True,
        )
        distance_features = self.distance_embedding(
            distance_squared,
            epsilon=epsilon,
        )
        edge_inputs = torch.cat(
            [
                receiver_node_latents,
                sender_node_latents,
                edge_latents,
                distance_features,
            ],
            dim=-1,
        )
        edge_latents = edge_latents + self.edge_mlp(edge_inputs)

        num_nodes = node_latents.shape[0]
        aggregated_messages = edge_latents.new_zeros(
            num_nodes,
            edge_latents.shape[-1],
        )
        aggregated_messages.index_add_(
            dim=0,
            index=receivers,
            source=edge_latents,
        )
        sum_update = self.node_mlp(
            torch.cat([norm_node_latents, aggregated_messages], dim=-1)
        )

        node_attention_list = []
        for node_idx in range(num_nodes):
            edge_mask = receivers == node_idx
            neighbor_latents = norm_node_latents[senders[edge_mask]]
            receiver_latents = norm_node_latents[node_idx].unsqueeze(0).expand_as(
                neighbor_latents
            )
            node_attention_list.append(
                self.multi_attention(
                    receiver_latents,
                    neighbor_latents,
                    edge_latents[edge_mask],
                )
            )

        attentions = torch.stack(node_attention_list, dim=0)
        attention_update = self.attention_mlp(
            torch.cat([norm_node_latents, attentions], dim=-1)
        )
        updated_nodes = (
            node_latents
            + sum_update
            + self.attention_scale * attention_update
        )
        node_latents = updated_nodes + self.ffn(self.norm2(updated_nodes))
        return node_latents, edge_latents


class GenericPotentialReadout(nn.Module):
    """Invariant node, symmetric-pair, and global potential readout."""

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        latent_dim,
        hidden_dim,
        distance_dim=16,
        num_hidden_layers=2,
        rbf_initial_scale=1.0,
        global_gate_init=1e-2,
    ):
        super().__init__()
        if global_gate_init < 0.0:
            raise ValueError("global_gate_init must be nonnegative")

        self.norm1 = nn.LayerNorm(latent_dim)
        self.distance_embedding = LearnableRadialBasis(
            num_basis=distance_dim,
            initial_scale=rbf_initial_scale,
        )
        self.edge_mlp = MLP(
            input_dim=latent_dim * 3 + distance_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )

        self.potential_node_mlp = MLP(
            input_dim=latent_dim + node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )
        directed_pair_input_dim = (
            latent_dim * 3
            + distance_dim
            + node_input_dim * 2
            + edge_input_dim
        )
        self.potential_edge_mlp = MLP(
            input_dim=directed_pair_input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )

        global_input_dim = (
            latent_dim * 2 + node_input_dim + edge_input_dim + 2
        )
        self.potential_global_mlp = MLP(
            input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )

        self.node_potential_scale = nn.Parameter(torch.tensor(1.0))
        self.pair_potential_scale = nn.Parameter(torch.tensor(1.0))
        self.global_potential_scale = nn.Parameter(
            torch.tensor(float(global_gate_init))
        )

    def forward(
        self,
        node_latents,
        edge_latents,
        node_features,
        edge_features,
        coordinates,
        senders,
        receivers,
        epsilon,
    ):
        norm_node_latents = self.norm1(node_latents)
        sender_latents = norm_node_latents[senders]
        receiver_latents = norm_node_latents[receivers]

        relative_position = coordinates[senders] - coordinates[receivers]
        distance_squared = relative_position.square().sum(
            dim=-1,
            keepdim=True,
        )
        distance_features = self.distance_embedding(
            distance_squared,
            epsilon=epsilon,
        )
        edge_inputs = torch.cat(
            [
                receiver_latents,
                sender_latents,
                edge_latents,
                distance_features,
            ],
            dim=-1,
        )
        final_messages = edge_latents + self.edge_mlp(edge_inputs)

        node_potential = self.potential_node_mlp(
            torch.cat([norm_node_latents, node_features], dim=-1)
        ).sum()

        directed_pair_inputs = torch.cat(
            [
                receiver_latents,
                sender_latents,
                final_messages,
                distance_features,
                node_features[receivers],
                node_features[senders],
                edge_features,
            ],
            dim=-1,
        )
        directed_pair_potentials = self.potential_edge_mlp(
            directed_pair_inputs
        )

        lower = torch.minimum(senders, receivers)
        upper = torch.maximum(senders, receivers)
        valid = lower != upper
        valid_pair_potentials = directed_pair_potentials[valid]

        if valid_pair_potentials.shape[0] > 0:
            pair_keys = (
                lower[valid] * node_latents.shape[0] + upper[valid]
            )
            unique_pairs, pair_indices = torch.unique(
                pair_keys,
                return_inverse=True,
            )
            pair_sums = directed_pair_potentials.new_zeros(
                unique_pairs.numel(),
                1,
            )
            pair_counts = directed_pair_potentials.new_zeros(
                unique_pairs.numel(),
                1,
            )
            pair_sums.index_add_(
                0,
                pair_indices,
                valid_pair_potentials,
            )
            pair_counts.index_add_(
                0,
                pair_indices,
                valid_pair_potentials.new_ones(
                    valid_pair_potentials.shape[0],
                    1,
                ),
            )
            pair_potential = (pair_sums / pair_counts).sum()
        else:
            pair_potential = 0.0 * coordinates.sum()

        if final_messages.shape[0] == 0:
            pooled_edge_latents = norm_node_latents.new_zeros(
                norm_node_latents.shape[-1]
            )
            pooled_edge_features = edge_features.new_zeros(
                edge_features.shape[-1]
            )
        else:
            pooled_edge_latents = final_messages.mean(dim=0)
            pooled_edge_features = edge_features.mean(dim=0)

        count_features = norm_node_latents.new_tensor(
            [node_latents.shape[0], edge_latents.shape[0]]
        ).log1p()
        global_inputs = torch.cat(
            [
                norm_node_latents.mean(dim=0),
                pooled_edge_latents,
                node_features.mean(dim=0),
                pooled_edge_features,
                count_features,
            ],
            dim=-1,
        )
        global_potential = self.potential_global_mlp(
            global_inputs.unsqueeze(0)
        ).squeeze()

        return (
            self.node_potential_scale * node_potential
            + self.pair_potential_scale * pair_potential
            + self.global_potential_scale * global_potential
        )


class GenericAblationEncodeProcessDecode(nn.Module):
    """SETNODE force model using the four law-agnostic ablations."""

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        latent_dim=128,
        hidden_dim=128,
        num_message_passing_steps=10,
        num_hidden_layers=2,
        node_mean=None,
        node_std=None,
        edge_mean=None,
        edge_std=None,
        num_heads=4,
        distance_dim=16,
        ffn_dim=256,
        epsilon=0.15,
        rbf_initial_scale=1.0,
        global_gate_init=1e-2,
    ):
        super().__init__()

        if num_message_passing_steps < 1:
            raise ValueError("num_message_passing_steps must be at least 1")
        if epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")

        self.epsilon = float(epsilon)

        def make_std_safe(std, size):
            if std is None:
                std = torch.ones(size)
            return torch.where(
                std.abs() < 1e-8,
                torch.ones_like(std),
                std,
            )

        if node_mean is None:
            node_mean = torch.zeros(node_input_dim)
        if edge_mean is None:
            edge_mean = torch.zeros(edge_input_dim)

        self.register_buffer("node_mean", node_mean)
        self.register_buffer(
            "node_std",
            make_std_safe(node_std, node_input_dim),
        )
        self.register_buffer("edge_mean", edge_mean)
        self.register_buffer(
            "edge_std",
            make_std_safe(edge_std, edge_input_dim),
        )

        self.node_encoder = MLP(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )
        self.edge_encoder = MLP(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )
        self.processors = nn.ModuleList([
            GenericInteractionNetwork(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                distance_dim=distance_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                rbf_initial_scale=rbf_initial_scale,
            )
            for _ in range(num_message_passing_steps)
        ])
        self.potential_readout = GenericPotentialReadout(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            distance_dim=distance_dim,
            num_hidden_layers=num_hidden_layers,
            rbf_initial_scale=rbf_initial_scale,
            global_gate_init=global_gate_init,
        )

    def forward(self, graph, create_graph=None):
        if create_graph is None:
            create_graph = self.training

        with torch.enable_grad():
            node_features = graph["node_features"]
            edge_features = graph["edge_features"]
            coordinates = graph["coordinates"]
            if not coordinates.requires_grad:
                coordinates = coordinates.detach().clone().requires_grad_(True)
            senders = graph["senders"]
            receivers = graph["receivers"]

            node_features = (node_features - self.node_mean) / self.node_std
            edge_features = (edge_features - self.edge_mean) / self.edge_std

            node_latents = self.node_encoder(node_features)
            edge_latents = self.edge_encoder(edge_features)
            for processor in self.processors:
                node_latents, edge_latents = processor(
                    node_latents=node_latents,
                    edge_latents=edge_latents,
                    coordinates=coordinates,
                    senders=senders,
                    receivers=receivers,
                    epsilon=self.epsilon,
                )

            potential = self.potential_readout(
                node_latents=node_latents,
                edge_latents=edge_latents,
                node_features=node_features,
                edge_features=edge_features,
                coordinates=coordinates,
                senders=senders,
                receivers=receivers,
                epsilon=self.epsilon,
            )
            force = -torch.autograd.grad(
                potential,
                coordinates,
                create_graph=create_graph,
            )[0]

        return force
