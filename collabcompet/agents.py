"""The agents used during the development of this project.

@{link Agent} is the basic DDPG agent as was used in the continuous project.

@{link IndependentAgent} consists of two DDPG agents acting independently.

@{link JointCriticAgent} consists of two DDPG agents sharing a critic.
"""
import logging
import random

import numpy as np
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
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def files_exist(self) -> bool:
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


class Agent(AgentInterface):
    def __init__(self, replay_memory_size, actor_count, state_size, action_size, run_id, numpy_seed=36, random_seed=21,
                 torch_seed=42):
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

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = self.actor_local.get_copy()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = self.critic_local.get_copy()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(self.action_size)

        self.step_count = 0
        self.update_every = UPDATE_EVERY

    def reset(self, idx=None):
        """Reset the agent.

        In particular we reset the noise process; when passed an integer the noise is scaled down proportional to that
        integer.  Otherwise it is just a restart of the noise process.
        """
        if idx:
            self.noise = OUNoise(self.action_size, mu=0.0, theta=1 / (idx + 2), sigma=1 / (idx + 2))
        else:
            self.noise.reset()

    def record_experience(self, experience: Experience):
        self.experiences.add(experience)
        self.step_count += 1
        if len(self.experiences) > BATCH_SIZE and self.step_count % self.update_every == 0:
            log.debug("Doing a learning step")
            self._learn()

    def get_action(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        assert state.shape == (24,)
        self.actor_local.eval()
        with torch.no_grad():
            # noinspection PyUnresolvedReferences
            action = self.actor_local(torch.from_numpy(state).float().to(device)).cpu().numpy()
        if add_noise:
            n = self.noise.sample()
            assert n.shape == (2,)
            log.debug("noise %s", n)
            action += n
        assert action.shape == (2,)
        return np.clip(action, -1, 1)

    def files_exist(self) -> bool:
        """Check if the model files already exist.

        Note: also checks that if any exist all exists.
        """
        f1 = os.path.isfile("{dir}trained_model-actor_local-{id}.pth".format(dir=DATA_DIR, id=self.run_id))
        f2 = os.path.isfile("{dir}trained_model-actor_target-{id}.pth".format(dir=DATA_DIR, id=self.run_id))
        f3 = os.path.isfile("{dir}trained_model-critic_local-{id}.pth".format(dir=DATA_DIR, id=self.run_id))
        f4 = os.path.isfile("{dir}trained_model-critic_target-{id}.pth".format(dir=DATA_DIR, id=self.run_id))
        all_files = np.all([f1, f2, f3, f4])
        any_files = np.any([f1, f2, f3, f4])
        if any_files:
            assert all_files
        return all_files

    def save(self) -> None:
        torch.save(self.actor_local.state_dict(), "{dir}trained_model-actor_local-{id}.pth"
                   .format(dir=DATA_DIR, id=self.run_id))
        torch.save(self.actor_target.state_dict(), "{dir}trained_model-actor_target-{id}.pth"
                   .format(dir=DATA_DIR, id=self.run_id))
        torch.save(self.critic_local.state_dict(), "{dir}trained_model-critic_local-{id}.pth"
                   .format(dir=DATA_DIR, id=self.run_id))
        torch.save(self.critic_target.state_dict(), "{dir}trained_model-critic_target-{id}.pth"
                   .format(dir=DATA_DIR, id=self.run_id))

    def load(self) -> None:
        self.actor_local.load_state_dict(torch.load("{dir}trained_model-actor_local-{id}.pth"
                                                    .format(dir=DATA_DIR, id=self.run_id)))
        self.actor_target.load_state_dict(torch.load("{dir}trained_model-actor_target-{id}.pth"
                                                     .format(dir=DATA_DIR, id=self.run_id)))
        self.critic_local.load_state_dict(torch.load("{dir}trained_model-critic_local-{id}.pth"
                                                     .format(dir=DATA_DIR, id=self.run_id)))
        self.critic_target.load_state_dict(torch.load("{dir}trained_model-critic_target-{id}.pth"
                                                      .format(dir=DATA_DIR, id=self.run_id)))

    def _learn(self):
        gamma = GAMMA
        self.actor_local.train()  # the other models are never switched out of train mode
        states, actions, rewards, next_states, dones = self.experiences.sample()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        Agent._soft_update(self.critic_local, self.critic_target, TAU)

        Agent._soft_update(self.actor_local, self.actor_target, TAU)


class IndependentAgent(AgentInterface):
    """Run the training with two completely independent DDPG agents.
    """

    def __init__(self, run_id):
        self.agent_A = Agent(replay_memory_size=100000, state_size=24, action_size=2, actor_count=1,
                             run_id=f"agent-A-{run_id}")
        self.agent_B = Agent(replay_memory_size=100000, state_size=24, action_size=2, actor_count=1,
                             run_id=f"agent-B-{run_id}")

    def reset(self, idx=None):
        self.agent_A.reset()
        self.agent_B.reset()

    def record_experience(self, experience: Experience):
        experience_a = Experience(experience.state[0, :],
                                  experience.action[0, :],
                                  experience.reward[0],
                                  experience.next_state[0, :],
                                  experience.done[0])
        self.agent_A.record_experience(experience_a)
        experience_b = Experience(experience.state[1, :],
                                  experience.action[1, :],
                                  experience.reward[1],
                                  experience.next_state[1, :],
                                  experience.done[1])
        self.agent_B.record_experience(experience_b)

    def get_action(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        action_a = self.agent_A.get_action(state[0, :])
        action_b = self.agent_B.get_action(state[1, :])
        return np.vstack([action_a, action_b])

    def save(self) -> None:
        self.agent_A.save()
        self.agent_B.save()

    def load(self) -> None:
        assert self.agent_A.files_exist() and self.agent_B.files_exist()
        self.agent_A.load()
        self.agent_B.load()

    def files_exist(self) -> bool:
        """Check if the model files exit.

        Since in the normal course of use of this code either both or neither files exists we check for the disjunction
        of both files_exists.
        """
        return self.agent_A.files_exist() or self.agent_B.files_exist()


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
        self.actor_1_local = Actor(state_size, action_size).to(device)
        self.actor_1_target = self.actor_1_local.get_copy()
        self.actor_1_optimizer = optim.Adam(self.actor_1_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        log.info("Actor1 local: %s", repr(self.actor_1_local))
        log.info("Actor1 target: %s", repr(self.actor_1_target))

        # Second actor network
        self.actor_2_local = Actor(state_size, action_size).to(device)
        self.actor_2_target = self.actor_2_local.get_copy()
        self.actor_2_optimizer = optim.Adam(self.actor_2_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        log.info("Actor2 local: %s", repr(self.actor_2_local))
        log.info("Actor2 target: %s", repr(self.actor_2_target))
        
        # Critic Network
        # note gets the actions and observations from both actors and hence has sizes twice as large
        self.critic_local = Critic(2 * state_size, 2 * action_size, agent_count=self.actor_count).to(device)
        self.critic_target = self.critic_local.get_copy()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        log.info("Critic local: %s", repr(self.critic_local))
        log.info("Critic target: %s", repr(self.critic_target))

        # Noise process
        self.noise = OUNoise((self.actor_count, self.action_size))

        self.step_count = 0
        self.update_every = UPDATE_EVERY

        self.actor_1_local_filename = "{dir}trained_model-actor_local_1-{id}.pth".format(dir=DATA_DIR, id=self.run_id)
        self.actor_1_target_filename = "{dir}trained_model-actor_target_1-{id}.pth".format(dir=DATA_DIR, id=self.run_id)
        self.actor_2_local_filename = "{dir}trained_model-actor_local_2-{id}.pth".format(dir=DATA_DIR, id=self.run_id)
        self.actor_2_target_filename = "{dir}trained_model-actor_target_2-{id}.pth".format(dir=DATA_DIR, id=self.run_id)
        self.critic_local_filename = "{dir}trained_model-critic_local-{id}.pth".format(dir=DATA_DIR, id=self.run_id)
        self.critic_target_filename = "{dir}trained_model-critic_target-{id}.pth".format(dir=DATA_DIR, id=self.run_id)

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

    def save(self) -> None:
        torch.save(self.actor_1_local.state_dict(), self.actor_1_local_filename)
        torch.save(self.actor_1_target.state_dict(), self.actor_1_target_filename)
        torch.save(self.actor_2_local.state_dict(), self.actor_2_local_filename)
        torch.save(self.actor_2_target.state_dict(), self.actor_2_target_filename)
        torch.save(self.critic_local.state_dict(), self.critic_local_filename)
        torch.save(self.critic_target.state_dict(), self.critic_target_filename)

    def load(self) -> None:
        self.actor_1_local.load_state_dict(torch.load(self.actor_1_local_filename))
        self.actor_1_target.load_state_dict(torch.load(self.actor_1_target_filename))
        self.actor_2_local.load_state_dict(torch.load(self.actor_2_local_filename))
        self.actor_2_target.load_state_dict(torch.load(self.actor_2_target_filename))
        self.critic_local.load_state_dict(torch.load(self.critic_local_filename))
        self.critic_target.load_state_dict(torch.load(self.critic_target_filename))

    def files_exist(self) -> bool:
        """Check if the model files already exist.

        Note: also checks that if any exist all exists.
        """
        f1 = os.path.isfile(self.actor_1_local_filename)
        f2 = os.path.isfile(self.actor_1_target_filename)
        f3 = os.path.isfile(self.actor_2_local_filename)
        f4 = os.path.isfile(self.actor_2_target_filename)
        f5 = os.path.isfile(self.critic_local_filename)
        f6 = os.path.isfile(self.critic_target_filename)
        all_files = np.all([f1, f2, f3, f4, f5, f6])
        any_files = np.any([f1, f2, f3, f4, f5, f6])
        if any_files:
            assert all_files
        return all_files

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
        Agent._soft_update(self.critic_local, self.critic_target, TAU)
        Agent._soft_update(self.actor_1_local, self.actor_1_target, TAU)
        Agent._soft_update(self.actor_2_local, self.actor_2_target, TAU)
