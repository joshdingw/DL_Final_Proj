from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.functional as F


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

class Encoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, repr_dim),  # Adjusted in_features to 128 * 9 * 9 = 10368
            nn.ReLU()
        )


    def forward(self, x):
        return self.conv_net(x)


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim)
        )

    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        return self.mlp(x)


class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, momentum=0.999):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.momentum = momentum

        self.encoder = Encoder(repr_dim).to(device)
        self.target_encoder = Encoder(repr_dim).to(device)
        self.predictor = Predictor(repr_dim, action_dim).to(device)

        # Initialize target encoder parameters with encoder parameters
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Stop gradients for target encoder

    @torch.no_grad()
    def update_target_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, C, H, W]
            actions: [B, T-1, action_dim]

        Returns:
            predictions: [B, T-1, repr_dim]
            target_reprs: [B, T-1, repr_dim]
        """
        B, T, C, H, W = states.shape

        # Encode all states with encoder and target encoder
        states_flat = states.view(B * T, C, H, W)
        state_reprs = self.encoder(states_flat)
        state_reprs = state_reprs.view(B, T, -1)  # [B, T, repr_dim]

        with torch.no_grad():
            target_state_reprs = self.target_encoder(states_flat)
            target_state_reprs = target_state_reprs.view(B, T, -1)  # [B, T, repr_dim]

        predictions = []
        for t in range(T - 1):
            pred = self.predictor(state_reprs[:, t], actions[:, t])
            predictions.append(pred.unsqueeze(1))

        predictions = torch.cat(predictions, dim=1)  # [B, T-1, repr_dim]
        target_reprs = target_state_reprs[:, 1:]     # [B, T-1, repr_dim]

        return predictions, target_reprs

    def predict_future(self, init_state, actions):
        """
        Unroll the model to predict future representations.

        Args:
            init_state: [B, 1, C, H, W]
            actions: [B, T-1, action_dim]

        Returns:
            predicted_reprs: [T, B, repr_dim]
        """
        B, _, C, H, W = init_state.shape
        T_minus1 = actions.shape[1]
        T = T_minus1 + 1

        predicted_reprs = []

        # Get representation of initial state
        state_repr = self.encoder(init_state.squeeze(1))  # [B, repr_dim]
        predicted_reprs.append(state_repr.unsqueeze(0))  # [1, B, repr_dim]

        for t in range(T_minus1):
            action = actions[:, t]  # [B, action_dim]
            # Predict next representation
            pred_repr = self.predictor(state_repr, action)
            predicted_reprs.append(pred_repr.unsqueeze(0))  # [1, B, repr_dim]
            # Update state representation for next step
            state_repr = pred_repr

        predicted_reprs = torch.cat(predicted_reprs, dim=0)  # [T, B, repr_dim]
        return predicted_reprs
