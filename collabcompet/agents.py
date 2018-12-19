"""The agents used during the development of this project.

@{link Agent} is the basic DDPG agent.

@{link IndependentAgent} consists of two DDPG agents acting independently.

@{link JointCriticAgent} consists of two DDPG agents sharing a critic.
"""
import logging.config
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim
import os.path

from abc import ABC, abstractmethod
from collabcompet.experiences import Experience, Experiences
from collabcompet.model import Actor, Critic
from collabcompet.noise import OUNoise

# noinspection PyUnresolvedReferences
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 1.0  # discount factor
TAU = 1e-2  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0.5  # L2 weight decay
UPDATE_EVERY = 1  # do a learning update after this many recorded experiences

DATA_DIR = "data/"


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
        all = np.all([f1, f2, f3, f4])
        any = np.any([f1, f2, f3, f4])
        if any:
            assert all
        return all

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
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
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
        experience_A = Experience(experience.state[0, :],
                                  experience.action[0, :],
                                  experience.reward[0],
                                  experience.next_state[0, :],
                                  experience.done[0])
        self.agent_A.record_experience(experience_A)
        experience_B = Experience(experience.state[1, :],
                                  experience.action[1, :],
                                  experience.reward[1],
                                  experience.next_state[1, :],
                                  experience.done[1])
        self.agent_B.record_experience(experience_B)

    def get_action(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        action_A = self.agent_A.get_action(state[0, :])
        action_B = self.agent_B.get_action(state[1, :])
        return np.vstack([action_A, action_B])

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