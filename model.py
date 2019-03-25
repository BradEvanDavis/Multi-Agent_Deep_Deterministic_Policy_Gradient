import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vars import Options
from vars import HyperParameters as Params


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc_units1=256, fc_units2=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(Options.seed_num)

        self.fc1 = nn.Linear(state_size, fc_units1)
        self.fc2 = nn.Linear(fc_units1, fc_units2)
        self.fc3 = nn.Linear(fc_units2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        out = F.selu(self.fc1(state))
        out = F.selu(self.fc2(out))
        return torch.tanh(self.fc3(out))

    def soft_update(self, local_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            TAU: interpolation parameter
        """
        for target_param, local_param in zip(self.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(Options.seed_num)
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """

        self.fc_1 = nn.Linear((state_size+action_size)*2, fc1_units)
        self.BatchNorm1b = nn.BatchNorm1d(fc1_units)
        self.fc_2 = nn.Linear(fc1_units, fc2_units)
        self.fc_3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        out = F.selu(self.BatchNorm1b(self.fc_1(x)))
        out = F.selu(self.fc_2(out))
        return self.fc_3(out)

    def soft_update(self, local_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            TAU: interpolation parameter
        """
        for target_param, local_param in zip(self.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

