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
        node decoder
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

        edge_input_dim = latent_dim * 3

        self.edge_mlp = MLP(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=True,
        )

        # Node model seeS:
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

    def forward(self, node_latents, edge_latents, senders, receivers):
        """
        Parameters:
            node_latents : torch.Tensor
                Shape : [num_nodes, latent_dim]

            edge_latents : torch.Tensor
                Shape : [num_edges, latent_dim]

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

        sender_node_latents = node_latents[senders]
        receiever_node_latents = node_latents[receivers]

        edge_inputs = torch.cat(
            [sender_node_latents, receiever_node_latents, edge_latents], dim=-1
        )

        edge_updates = self.edge_mlp(edge_inputs)

        # Residual edge update, this is used so that we dont replace the old information with new information but instead update it by adding the learned update to the previous edge latent
        edge_latents = edge_updates + edge_latents

        # Aggegrate incoming edge messages for each reciever node.
        num_nodes = node_latents.shape[0]

        # torch.zeros(a,b) creates a tensor of zeros with dimensions a, b.
        aggregated_messages = torch.zeros(
            num_nodes,
            edge_latents.shape[-1],
            device=node_latents.device,
            dtype=node_latents.dtype,
        )

        aggregated_messages.index_add_(dim=0, index=receivers, source=edge_latents)

        node_inputs = torch.cat([node_latents, aggregated_messages], dim=-1)

        node_updates = self.node_mlp(node_inputs)

        node_latents = node_updates + node_latents

        return node_latents, edge_latents


class EncodeProcessDecode(nn.Module):
    """
    Graph Network Simulator model.

    """

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        output_dim,
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

        self.register_buffer("node_mean", node_mean)
        self.register_buffer("node_std", node_std)
        self.register_buffer("edge_mean", edge_mean)
        self.register_buffer("edge_std", edge_std)

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

        # Decoder predicts acceleration from node latent features.

        self.node_decoder = MLP(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            use_layer_norm=False,
        )

    def forward(self, graph):
        node_features = graph["node_features"]
        edge_features = graph["edge_features"]
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
                senders=senders,
                receivers=receivers,
            )

        predicted_acceleration = self.node_decoder(node_latents)

        return predicted_acceleration
