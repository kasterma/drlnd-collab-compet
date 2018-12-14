import logging.config
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim

from collabcompet.experience import Experience, Experiences
from collabcompet.model import Actor, Critic
from collabcompet.noise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # do a learning update after this many recorded experiences


class Agent:
    def __init__(self, replay_memory_size, actor_count, state_size, action_size, numpy_seed=36, random_seed=21, torch_seed=42):
        log.info("Random seeds, numpy %d, random %d, torch %d.", numpy_seed, random_seed, torch_seed)
        # seed all sources of randomness
        torch.manual_seed(torch_seed)
        # noinspection PyUnresolvedReferences
        np.random.seed(seed=numpy_seed)
        random.seed(random_seed)

        self.experiences = Experiences(memory_size=replay_memory_size, batch_size=BATCH_SIZE)

        self.actor_count = actor_count
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = self.actor_local.get_copy()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

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
            self.noise = OUNoise(self.action_size, mu=0.0, theta=1/(idx + 2), sigma=1/(idx + 2))
        else:
            self.noise.reset()

    def record_experience(self, experience: Experience):
        self.experiences.add(experience)
        self.step_count += 1
        if len(self.experiences) > BATCH_SIZE and self.step_count % self.update_every == 0:
            log.debug("Doing a learning step")
            self._learn()

    def get_action(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(torch.from_numpy(state).float().to(device)).cpu().numpy()
        if add_noise:
            action += self.noise.sample()
        return action

    def save(self, run_identifier: str) -> None:
        torch.save(self.actor_local.state_dict(), "trained_model-actor_local-{id}.pth".format(id=run_identifier))
        torch.save(self.actor_target.state_dict(), "trained_model-actor_target-{id}.pth".format(id=run_identifier))
        torch.save(self.critic_local.state_dict(), "trained_model-critic_local-{id}.pth".format(id=run_identifier))
        torch.save(self.critic_target.state_dict(), "trained_model-critic_target-{id}.pth".format(id=run_identifier))

    def load(self, run_identifier: str) -> None:
        self.actor_local.load_state_dict(torch.load("trained_model-actor_local-{id}.pth".format(id=run_identifier)))
        self.actor_target.load_state_dict(torch.load("trained_model-actor_target-{id}.pth".format(id=run_identifier)))
        self.critic_local.load_state_dict(torch.load("trained_model-critic_local-{id}.pth".format(id=run_identifier)))
        self.critic_target.load_state_dict(torch.load("trained_model-critic_target-{id}.pth".format(id=run_identifier)))

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

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        """Move the weights from the target_model in the direction of the local_model.

        The parameter tau determines how far this move is, we take a convex combination of the target and local weights
        where as tau increases we take the combination closer to the local parameters (tau equal to 1 would replace
        the target model with the local model, tau equal to 0 would perform no update and leave the target model as it
        is).
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
