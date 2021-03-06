"""The agents used during the development of this project.

@{link Agent} is the basic DDPG agent as was used in the continuous project.

@{link IndependentAgent} consists of two DDPG agents acting independently.

@{link JointCriticAgent} consists of two DDPG agents sharing a critic.
"""
import logging
import random

import numpy as np
import pandas as pd
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
import yaml
from torch import optim
import os.path

from abc import ABC, abstractmethod
from collabcompet.experiences import Experience, Experiences
from collabcompet.model import Actor, Critic
from collabcompet.noise import OUNoise
from collabcompet.config import device, config

log = logging.getLogger("agent")

BUFFER_SIZE = config['buffer_size']
BATCH_SIZE = config['batch_size']
GAMMA = float(config['gamma'])
TAU = float(config['tau'])
LR_ACTOR = float(config['learning_rate_actor'])
LR_CRITIC = float(config['learning_rate_critic'])
WEIGHT_DECAY = config['weight_decay']
UPDATE_EVERY = config['update_every']

DATA_DIR = config['data_dir']


class AgentInterface(ABC):
    @abstractmethod
    def reset(self, idx=None):
        pass

    @abstractmethod
    def record_experience(self, experience: Experience):
        pass

    @abstractmethod
    def get_action(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        pass

    @abstractmethod
    def save(self,  episode_idx: int, label: str = "") -> None:
        """Save with the label added to the filename"""
        pass

    @abstractmethod
    def load(self, run_id: int,  episode_idx: int, label: str = "") -> None:
        """Load with the label added to the filename"""
        pass

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        """Move the weights from the target_model in the direction of the local_model.

        The parameter tau determines how far this move is, we take a convex combination of the target and local weights
        where as tau increases we take the combination closer to the local parameters (tau equal to 1 would replace
        the target model with the local model, tau equal to 0 would perform no update and leave the target model as it
        is).
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MADDPG(AgentInterface):
    """Multi agent deep deterministic policy gradient.
    """

    def __init__(self, replay_memory_size, actor_count, state_size, action_size, run_id, numpy_seed=36, random_seed=21,
                 torch_seed=42):
        """

        :param replay_memory_size:
        :param state_size: observation space size for a single agent
        :param action_size: action space size for a single agent
        :param run_id:
        :param numpy_seed:
        :param random_seed:
        :param torch_seed:
        """
        log.info("Random seeds, numpy %d, random %d, torch %d.", numpy_seed, random_seed, torch_seed)
        # seed all sources of randomness
        torch.manual_seed(torch_seed)
        # noinspection PyUnresolvedReferences
        np.random.seed(seed=numpy_seed)
        random.seed(random_seed)

        self.experiences = Experiences(memory_size=replay_memory_size, batch_size=BATCH_SIZE)

        self.run_id = run_id
        self.actor_count = actor_count
        self.state_size = state_size
        self.action_size = action_size

        # First actor Network
        self.actor_1_local = Actor(state_size, action_size, model_label=f"actor_1_local-run_{run_id}").to(device)
        self.actor_1_target = self.actor_1_local.get_copy(model_label=f"actor_1_target-run_{run_id}")
        self.actor_1_optimizer = optim.Adam(self.actor_1_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        log.info("Actor1 local: %s", repr(self.actor_1_local))
        log.info("Actor1 target: %s", repr(self.actor_1_target))

        # Second actor network
        self.actor_2_local = Actor(state_size, action_size, model_label=f"actor_2_local-run_{run_id}").to(device)
        self.actor_2_target = self.actor_2_local.get_copy(model_label=f"actor_2_target-run_{run_id}")
        self.actor_2_optimizer = optim.Adam(self.actor_2_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        log.info("Actor2 local: %s", repr(self.actor_2_local))
        log.info("Actor2 target: %s", repr(self.actor_2_target))

        # Critic Network
        # note gets the actions and observations from both actors and hence has sizes twice as large
        self.critic_local = Critic(2 * state_size, 2 * action_size, agent_count=self.actor_count,
                                   model_label=f"critic_local-run_{run_id}").to(device)
        self.critic_target = self.critic_local.get_copy(model_label=f"critic_target-run_{run_id}")
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        log.info("Critic local: %s", repr(self.critic_local))
        log.info("Critic target: %s", repr(self.critic_target))

        # Noise process
        self.noise = OUNoise((self.actor_count, self.action_size))

        self.step_count = 0
        self.update_every = UPDATE_EVERY

        self.loss = [0., 0., 0.]

    def reset(self, idx=None):
        self.noise.reset()

    def record_experience(self, experience: Experience):
        self.experiences.add(experience)
        self.step_count += 1
        if len(self.experiences) > BATCH_SIZE and self.step_count % self.update_every == 0:
            log.debug("Doing a learning step")
            self._learn()

    # noinspection PyUnresolvedReferences
    def get_action(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        assert state.shape == (self.actor_count, self.state_size)
        self.actor_1_local.eval()
        self.actor_2_local.eval()
        with torch.no_grad():
            action_1 = self.actor_1_local(torch.from_numpy(state[0, :]).float().to(device)).cpu().numpy()
            action_2 = self.actor_2_local(torch.from_numpy(state[1, :]).float().to(device)).cpu().numpy()
        actions = np.vstack([action_1, action_2])  # the environment expects the actions stacked
        assert actions.shape == (self.actor_count, self.action_size)
        if add_noise:
            actions += self.noise.sample()
        return actions

    def save(self, episode_idx: int, label: str = "") -> None:
        self.actor_1_local.save_to_db(episode_idx, label)
        self.actor_1_target.save_to_db(episode_idx, label)
        self.actor_2_local.save_to_db(episode_idx, label)
        self.actor_2_target.save_to_db(episode_idx, label)
        self.critic_local.save_to_db(episode_idx, label)
        self.critic_target.save_to_db(episode_idx, label)

    def load(self, run_id: int, episode_idx: int) -> 'MADDPG':
        self.actor_1_local = self.actor_1_local.read_from_db(run_id, "actor_1_local-run%", episode_idx)
        self.actor_1_target = self.actor_1_target.read_from_db(run_id, "actor_1_target-run%", episode_idx)
        self.actor_2_local = self.actor_2_local.read_from_db(run_id, "actor_2_local-run%", episode_idx)
        self.actor_2_target = self.actor_2_target.read_from_db(run_id, "actor_2_target-run%", episode_idx)
        self.critic_local = self.critic_local.read_from_db(run_id, "critic_local-run%", episode_idx)
        self.critic_target = self.critic_target.read_from_db(run_id, "critic_target-run%", episode_idx)
        return self

    def asDataFrame(self, episode_idx: int = -1):
        df_actor_1_local = pd.concat([pd.DataFrame({'label': k, 'value': v.detach().flatten()})
                                      for k, v in self.actor_1_local.named_parameters()])
        df_actor_1_local['net'] = 'actor_1_local'
        df_actor_1_target = pd.concat([pd.DataFrame({'label': k, 'value': v.detach().flatten()})
                                      for k, v in self.actor_1_target.named_parameters()])
        df_actor_1_target['net'] = 'actor_1_target'

        df = pd.concat([df_actor_1_local, df_actor_1_target])

        df['episode_idx'] = episode_idx

        return df

    # noinspection PyUnresolvedReferences
    def _learn(self):
        gamma = GAMMA
        self.actor_1_local.train()  # the critic model is never switched out of train mode
        self.actor_2_local.train()

        states, actions, rewards, next_states, dones = self.experiences.sample()

        assert states.shape == torch.Size([BATCH_SIZE, self.actor_count * self.state_size])
        assert actions.shape == torch.Size([BATCH_SIZE, self.actor_count * self.action_size])
        assert rewards.shape == torch.Size([BATCH_SIZE, self.actor_count])
        assert next_states.shape == torch.Size([BATCH_SIZE, self.actor_count * self.state_size])
        assert dones.shape == torch.Size([BATCH_SIZE, self.actor_count])

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Q: why not recorded from the actual run?  SARSA (second A are the recorded actions)
        action_1 = self.actor_1_local(next_states[:, :self.state_size])
        action_2 = self.actor_2_local(next_states[:, self.state_size:])
        actions_next = torch.cat([action_1, action_2], 1)
        assert actions_next.shape == (BATCH_SIZE, self.actor_count * self.action_size)

        q_targets_next = self.critic_target(next_states, actions_next)
        assert q_targets_next.shape == torch.Size((BATCH_SIZE, 2))
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        assert q_targets.shape == torch.Size((BATCH_SIZE, self.actor_count))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_1_pred = self.actor_1_local(states[:, :self.state_size])
        assert actions_1_pred.shape == (BATCH_SIZE, self.action_size)
        assert torch.cat([actions[:,:2], actions_1_pred], 1).shape == (BATCH_SIZE, self.action_size * 2)
        actor_1_loss = -self.critic_local(states, torch.cat([actions_1_pred, actions[:, 2:]], 1))[:, 0].mean()
        # Minimize the loss
        self.actor_1_optimizer.zero_grad()
        actor_1_loss.backward()
        self.actor_1_optimizer.step()

        # Compute actor loss
        actions_2_pred = self.actor_2_local(states[:, self.state_size:])
        actor_2_loss = -self.critic_local(states, torch.cat([actions[:, :2], actions_2_pred], 1))[:, 1].mean()
        # Minimize the loss
        self.actor_2_optimizer.zero_grad()
        actor_2_loss.backward()
        self.actor_2_optimizer.step()

        # ----------------------- update target networks ----------------------- #

        self._soft_update(self.critic_local, self.critic_target, TAU)
        self._soft_update(self.actor_1_local, self.actor_1_target, TAU)
        self._soft_update(self.actor_2_local, self.actor_2_target, TAU)

        # ------------------------- storing loss values ------------------------- #
        self.loss = [-actor_1_loss.item(), -actor_2_loss.item(), critic_loss.item()]

