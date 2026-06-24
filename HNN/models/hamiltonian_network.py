import torch
import torch.nn as nn


class HNN(nn.Module):
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(self, input_dim, hidden_dim, q_dim, state_mean=None, state_std=None):
        super().__init__()

        if state_mean is None:
            state_mean = torch.zeros(input_dim)

        if state_std is None:
            state_std = torch.ones(input_dim)

        # register_buffer stores the tensors with the model so they move with .to(device) and get saved in model.state_dict(). It also makes it so that they are not
        # trainable or that it will not be updated by the optimizer
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)

        self.q_dim = q_dim

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
        return self.net(x)

    def time_derivative(self, x):
        # Tells PyTorch to track operations on x so that we can take the derivative with respect to x
        x = x.requires_grad_(True)

        z = ((x - self.state_mean) / self.state_std).requires_grad_(True)
        H = self.forward(z)

        # H.sum() => H_1 + H_2 + ...
        # differentiate with respect to x/state
        # create computational graph so that we can backpropagate later
        # [0] because function returns a tuple (grad,)
        gradH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        qp_dim = self.q_dim * 2

        # because gradH is now computed with respect to normalized constants we must scale the derivatives also
        q_std = self.state_std[: self.q_dim]
        p_std = self.state_std[self.q_dim : qp_dim]
        # we need the derivative with respect to the state and not the normalized state so we divide by std according to the chain rule
        # dH / dx = dH / dz * dz/dx
        dH_dq = gradH[:, : self.q_dim] / q_std
        dH_dp = gradH[:, self.q_dim : qp_dim] / p_std

        q_dot = dH_dp
        p_dot = -dH_dq

        return torch.cat([q_dot, p_dot], dim=-1)
