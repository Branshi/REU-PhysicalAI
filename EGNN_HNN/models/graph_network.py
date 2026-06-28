import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Used for:
        node encoder
        edge encoder
        edge update function
        node update function
        coordinate weight function
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_hidden_layers=2,
        use_layer_norm=True,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        # *layers unpacks the list into seperate arguments.
        self.net = nn.Sequential(*layers)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.layer_norm(x)
        return x


class InteractionNetwork(nn.Module):
    """
    One message-passing block

    Updates:
        edge latent features
        node latent features
    """

    def __init__(self, latent_dim, hidden_dim, num_hidden_layers=2):
        super().__init__()

        # Edge model sees:
        # sender node latent
        # receiver node latent
        # current edge latent

        # + 1 for squared distance.
        edge_input_dim = latent_dim * 3 + 1

        self.edge_mlp = MLP(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )

        # Node model sees:
        # current node latent
        # aggegrated incoming edge messages

        node_input_dim = latent_dim * 2

        self.node_mlp = MLP(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )

    def forward(self, node_latents, edge_latents, coordinates, senders, receivers):
        """
        Parameters:
            node_latents : torch.Tensor
                Shape : [num_nodes, latent_dim]

            edge_latents : torch.Tensor
                Shape : [num_edges, latent_dim]

            coordinates : torch.Tensor
                      : [num_nodes, spatial_dim]

            senders : torch.Tensor
                Shape : [num_edges]

            receivers : torch.Tensor
                Shape : [num_edges]

        Returns:
            node_latents : torch.Tensor
                Shape : [num_nodes, latent_dim]

            edge_latents : torch.Tensor
                Shape : [num_edges, latent_dim]
        """

        receiver_node_latents = node_latents[receivers]
        sender_node_latents = node_latents[senders]

        relative_position = coordinates[senders] - coordinates[receivers]
        distance_squared = (relative_position**2).sum(dim=-1, keepdim=True)

        edge_inputs = torch.cat(
            [
                receiver_node_latents,
                sender_node_latents,
                distance_squared,
                edge_latents,
            ],
            dim=-1,
        )

        edge_updates = self.edge_mlp(edge_inputs)

        # Residual edge update, this is used so that we dont replace the old information with new information but instead update it by adding the learned update to the previous edge latent
        edge_latents = edge_updates + edge_latents

        # Aggegrate incoming edge messages for each reciever node.
        num_nodes = node_latents.shape[0]

        # torch.zeros(a,b) creates a tensor of zeros with dimensions a, b.
        # this piece of code is used the coreate a tensor with first dimension representing nodes and
        # second dimension representing the latent_dim = edge_latent.shape[-1].
        aggregated_messages = torch.zeros(
            num_nodes,
            # shape [num_edges, latent_dim]
            edge_latents.shape[-1],
            device=node_latents.device,
            dtype=node_latents.dtype,
        )

        # this computes the sum of edge latent vectors that are connected to a specific node
        aggregated_messages.index_add_(dim=0, index=receivers, source=edge_latents)

        node_inputs = torch.cat([node_latents, aggregated_messages], dim=-1)

        node_updates = self.node_mlp(node_inputs)

        node_latents = node_updates + node_latents

        return node_latents, edge_latents


class PotentialReadout(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_hidden_layers=2):
        super().__init__()

        self.edge_mlp = MLP(
            input_dim=latent_dim * 3 + 1,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )

        self.potential_edge_mlp = MLP(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )

        self.potential_node_mlp = MLP(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )

    def forward(
        self,
        node_latents,
        edge_latents,
        coordinates,
        senders,
        receivers,
    ):
        sender_latents = node_latents[senders]
        receiver_latents = node_latents[receivers]

        relative_position = coordinates[senders] - coordinates[receivers]
        distance_squared = relative_position.square().sum(
            dim=-1,
            keepdim=True,
        )

        edge_inputs = torch.cat(
            [
                receiver_latents,
                sender_latents,
                distance_squared,
                edge_latents,
            ],
            dim=-1,
        )

        final_messages = edge_latents + self.edge_mlp(edge_inputs)
        node_potential = self.potential_node_mlp(node_latents).sum()

        # Give (i, j) and (j, i) the same pair ID.
        # minimum and maximum order each pair without changing the graph edges.
        lower = torch.minimum(senders, receivers)
        upper = torch.maximum(senders, receivers)

        # Exclude self-edges.
        valid = lower != upper
        lower = lower[valid]
        upper = upper[valid]
        valid_messages = final_messages[valid]

        # If there are no valid pairs, return only the node contribution.
        if valid_messages.shape[0] == 0:
            # Keep an autograd connection to coordinates so the force is zero.
            return node_potential + 0.0 * coordinates.sum()

        # Convert each particle pair into a unique ID. Multiplying lower by
        # the number of nodes reserves N keys per lower index and avoids collisions.
        pair_keys = lower * node_latents.shape[0] + upper
        # Get the unique pairs.
        # suppose lower = [0, 0, 2], upper = [2, 2, 3], then pair_keys = [2, 2, 11] if N = 4.
        # return_inverse=True asks torch.unique() to return a mapping from every original element to its position in the unique result.
        # unique_pairs then becomes [2, 11], and pair_indices = [0, 0, 1].
        unique_pairs, pair_indices = torch.unique(pair_keys, return_inverse=True)
        num_pairs = unique_pairs.numel()

        pair_message_sums = final_messages.new_zeros(
            num_pairs,
            final_messages.shape[-1],
        )
        pair_counts = final_messages.new_zeros(num_pairs, 1)

        pair_message_sums.index_add_(0, pair_indices, valid_messages)
        pair_counts.index_add_(
            0,
            pair_indices,
            valid_messages.new_ones(valid_messages.shape[0], 1),
        )

        pair_messages = pair_message_sums / pair_counts
        pair_potential = self.potential_edge_mlp(pair_messages).sum()

        return node_potential + pair_potential


class EncodeProcessDecode(nn.Module):
    """
    EGNN-HNN potential model.

    """

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
    ):

        super().__init__()

        # A constant input feature has zero standard deviation, which would
        # cause division by zero. Replace near-zero standard deviations with 1.
        def make_std_safe(std, epsilon=1e-8):
            if std is None:
                return None

            return torch.where(
                std.abs() < epsilon,
                torch.ones_like(std),
                std,
            )

        if num_message_passing_steps < 1:
            raise ValueError("num_message_passing_steps must be at least 1")

        if node_mean is None:
            node_mean = torch.zeros(node_input_dim)
        if node_std is None:
            node_std = torch.ones(node_input_dim)
        if edge_mean is None:
            edge_mean = torch.zeros(edge_input_dim)
        if edge_std is None:
            edge_std = torch.ones(edge_input_dim)

        self.register_buffer("node_mean", node_mean)
        self.register_buffer("node_std", make_std_safe(node_std))
        self.register_buffer("edge_mean", edge_mean)
        self.register_buffer("edge_std", make_std_safe(edge_std))

        self.node_encoder = MLP(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )

        self.edge_encoder = MLP(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )

        self.processors = nn.ModuleList([
            InteractionNetwork(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            for _ in range(num_message_passing_steps)
        ])
        self.potential_readout = PotentialReadout(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, graph, create_graph=None):

        if create_graph is None:
            create_graph = self.training

        with torch.enable_grad():
            node_features = graph["node_features"]
            edge_features = graph["edge_features"]
            coordinates = graph["coordinates"]
            # in case of differentiable rollout, coordinates.requires_grad may be true and in the case we wish to preserve its history
            if not coordinates.requires_grad:
                # if requires_grad is false then we want to set it to true but not on the original coordinate tensor because
                # we only need it true temporiliy for autograd
                # thus we use detach to create a tensor with the same values but different autograd history
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
                )

            potential = self.potential_readout(
                node_latents=node_latents,
                edge_latents=edge_latents,
                coordinates=coordinates,
                senders=senders,
                receivers=receivers,
            )

            # Compute force/p_dot from potential
            # Force has the same shape as coordinates [num_nodes, spatial_dim].
            force = -torch.autograd.grad(
                potential, coordinates, create_graph=create_graph
            )[0]

        return force
