# The Actor and Critic models
#
# Essentially copied from the example code, but removed the "abuse" of the random seed.  The abuse was to reseed torch
# with the same seed on every model creation, hence if you create the same model twice you get identical generated
# random initialized values.  However cleaner to me is to create a get_copy function which makes the reuse of values
# explicit.

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F


def hidden_init(layer) -> Tuple[float, float]:
    fan_in = layer.weight.data.size()[0]
    # noinspection PyUnresolvedReferences
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=75, fc2_units=50):
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
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.tensor) -> torch.tensor:
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # noinspection PyUnresolvedReferences
        return torch.tanh(self.fc3(x))

    def get_copy(self) -> 'Actor':
        copy = Actor(self.state_size, self.action_size, self.fc1_units, self.fc2_units)
        for copy_param, self_param in zip(copy.parameters(), self.parameters()):
            copy_param.data.copy_(self_param.data)
        return copy


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, agent_count=1, fcs1_units=75, fc2_units=50):
        """
        :param state_size: size of the per agent state
        :param action_size: size of the per agent actions
        :param agent_count: number of agents to have critics for
        :param fcs1_units:
        :param fc2_units:
        """
        super(Critic, self).__init__()
        self.agent_count = agent_count
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fcs1_units
        self.fc2_units = fc2_units

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, self.agent_count)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        # noinspection PyUnresolvedReferences
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_copy(self) -> 'Critic':
        copy = Critic(self.state_size, self.action_size, self.agent_count, self.fc1_units, self.fc2_units)
        for copy_param, self_param in zip(copy.parameters(), self.parameters()):
            copy_param.data.copy_(self_param.data)
        return copy
