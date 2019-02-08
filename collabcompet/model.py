# The Actor and Critic models
#
# Essentially copied from the example code, but removed the "abuse" of the random seed.  The abuse was to reseed torch
# with the same seed on every model creation, hence if you create the same model twice you get identical generated
# random initialized values.  However cleaner to me is to create a get_copy function which makes the reuse of values
# explicit.
import sys
from typing import Tuple

import numpy as np
import torch
import os
import pickle
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

from collabcompet.config import config
import collabcompet.orm as orm
import logging

from abc import ABC

log = logging.getLogger('models')


def hidden_init(layer) -> Tuple[float, float]:
    fan_in = layer.weight.data.size()[0]
    # noinspection PyUnresolvedReferences
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class DBSave(ABC):
    def save_to_db(self, episode_idx: int, label: str) -> int:
        """Save the model to the database.

        Note: we expect the model to be uniquely identified by run_id, label and model_label; but the id is guaranteed
        to uniquely identify it.

        @:param label identifying label for this model

        @:return the id of the model in the database"""
        model_tosave = orm.Model(model_label=self.config['model_label'],
                                 run_id=orm.run.id, episode_idx=episode_idx, label=label,
                                 model_config=self.config, model_dict=self.state_dict())
        orm.session.add(model_tosave)
        orm.session.commit()
        return model_tosave.id

    def read_from_db(self, run_id: int, model_label: str, episode_idx: int) -> 'Actor':
        # TODO: fix return type
        try:
            model = orm.session.query(orm.Model) \
                .filter_by(run_id=run_id, episode_idx=episode_idx, model_label=model_label) \
                .one()
            a = self.__class__(**model.model_config)
            a.load_state_dict(model.model_dict)
            return a
        except (NoResultFound, MultipleResultsFound) as e:
            log.error(f"model not uniquely identified by {run_id}, {model_label}: {e}")
            sys.exit(1)


class Actor(nn.Module, DBSave):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=config['actor_fc1_units'],
                 fc2_units=config['actor_fc2_units'], model_label: str = "actor"):
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
        self.config = {'model_label': model_label,
                       'state_size': state_size,
                       'action_size': action_size,
                       'fc1_units': fc1_units,
                       'fc2_units': fc2_units}
        self.fc1 = nn.Linear(self.config['state_size'], self.config['fc1_units'])
        self.fc2 = nn.Linear(self.config['fc1_units'], self.config['fc2_units'])
        self.fc3 = nn.Linear(self.config['fc2_units'], self.config['action_size'])
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

    def get_copy(self, model_label: str) -> 'Actor':
        copy = Actor(self.config['state_size'], self.config['action_size'],
                     self.config['fc1_units'], self.config['fc2_units'],
                     model_label=model_label)
        for copy_param, self_param in zip(copy.parameters(), self.parameters()):
            copy_param.data.copy_(self_param.data)
        return copy

    def save(self, label: str, directory: str) -> None:
        """Save the model.

        Works together with load, when calling load with the same label as save the original state of the model will
        be restored.
        """
        torch.save(self.state_dict(), self._filename(label, directory))

    def load(self, label: str, directory: str) -> None:
        """Load the model

        Restore the state of the model from when save was called with the same label.
        """
        self.load_state_dict(torch.load(self._filename(label, directory)))

    def file_exists(self, label: str, directory: str):
        return os.path.isfile(self._filename(label, directory))

    def _filename(self, label: str, directory: str) -> str:
        sep = "-" if len(label) > 0 else ""
        return os.path.join(directory, f"{self.config['model_label']}{sep}{label}.pth")

    def __repr__(self):
        return f'Actor({self.config})'


class Critic(nn.Module, DBSave):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, agent_count=1,
                 fc1_units=config['critic_fc1_units'], fc2_units=config['critic_fc2_units'],
                 model_label: str = "critic"):
        """
        :param state_size: size of the per agent state
        :param action_size: size of the per agent actions
        :param agent_count: number of agents to have critics for
        :param fc1_units:
        :param fc2_units:
        """
        super(Critic, self).__init__()
        self.config = {'model_label': model_label,
                       'agent_count': agent_count,
                       'state_size': state_size,
                       'action_size': action_size,
                       'fc1_units': fc1_units,
                       'fc2_units': fc2_units}

        self.fcs1 = nn.Linear(self.config['state_size'], self.config['fc1_units'])
        self.fc2 = nn.Linear(self.config['fc1_units'] + self.config['action_size'], self.config['fc2_units'])
        self.fc3 = nn.Linear(self.config['fc2_units'], self.config['agent_count'])
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

    def get_copy(self, model_label: str) -> 'Critic':
        copy = Critic(self.config['state_size'], self.config['action_size'], self.config['agent_count'],
                      self.config['fc1_units'], self.config['fc2_units'],
                      model_label=model_label)
        for copy_param, self_param in zip(copy.parameters(), self.parameters()):
            copy_param.data.copy_(self_param.data)
        return copy

    def save(self, label: str, directory: str) -> None:
        """Save the model.

        Works together with load, when calling load with the same label as save the original state of the model will
        be restored.
        """
        torch.save(self.state_dict(), self._filename(label, directory))

    def load(self, label: str, directory: str) -> None:
        """Load the model

        Restore the state of the model from when save was called with the same label.
        """
        self.load_state_dict(torch.load(self._filename(label, directory)))

    def file_exists(self, label: str, directory: str):
        return os.path.isfile(self._filename(label, directory))

    def _filename(self, label: str, directory: str) -> str:
        sep = "-" if len(label) > 0 else ""
        return os.path.join(directory, f"{self.config['model_label']}{sep}{label}.pth")

    def __repr__(self):
        return f"Critic({self.config})"
