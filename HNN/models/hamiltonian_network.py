import torch
import torch.nn as nn


class HNN(nn.Module):
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(
        self, input_dim, hidden_dim, spatial_dim, state_mean=None, state_std=None
    ):
        super().__init__()

        if state_mean is None:
            state_mean = torch.zeros(input_dim)

        if state_std is None:
            state_std = torch.ones(input_dim)

        # register_buffer stores the tensors with the model so they move with .to(device) and get saved in model.state_dict(). It also makes it so that they are not
        # trainable or that it will not be updated by the optimizer
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std.clamp_min(1e-8))

        self.spatial_dim = spatial_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        z = (x - self.state_mean) / self.state_std
        return self.net(z)

    def force(self, x, create_graph=None):
        if create_graph is None:
            create_graph = self.training

        # Match EGNN_HNN: the network emits a scalar learned potential, while the
        # supervised output is force from -dV/dq. Enable gradients locally so eval
        # rollouts can call this from no_grad regions.
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().clone().requires_grad_(True)

            potential = self.forward(x)
            grad_potential = torch.autograd.grad(
                potential.sum(),
                x,
                create_graph=create_graph,
            )[0]

        return -grad_potential[..., : self.spatial_dim]
