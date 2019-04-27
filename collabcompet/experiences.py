import random
from collections import deque
from typing import Tuple, Union

import numpy as np
import torch
from collabcompet.config import device


class Experience:
    """Wrapper class for an experience as received from the environment"""

    def __init__(self, state: np.ndarray, action: np.ndarray, reward: Union[float, np.ndarray], next_state: np.ndarray,
                 done: bool,
                 joint=False):
        self.joint = joint

        if self.joint:
            assert state.shape == (2, 24)
            self.state = state.flatten()
        else:
            assert state.shape == (24,)
            self.state = state

        if self.joint:
            assert action.shape == (2, 2)
            self.action = action.flatten()
        else:
            assert action.shape == (2,)
            self.action = action

        if self.joint:
            assert type(reward) == list
            assert len(reward) == 2
            self.reward = reward
        else:
            self.reward = reward

        if self.joint:
            assert next_state.shape == (2, 24)
            self.next_state = next_state.flatten()
        else:
            assert next_state.shape == (24,)
            self.next_state = next_state

        self.done = done

    def __repr__(self):
        return "Experience(" + str(self.state) + "," + str(self.action) + "," + str(self.reward) + "," +\
               str(self.next_state) + "," + str(self.done) + ")"


class Experiences:
    """
    Fixed-size buffer to store experience tuples.

    Store the data as received from the Unity environment, but before returning the sample prepare it for use by the
    agent for learning (the agent can ignore that it seems to store numpy data and needs torch tensors).
    """

    def __init__(self, memory_size: int, batch_size: int):
        self.memory = deque(maxlen=memory_size)
        self.sample_size = batch_size      # in the current learning method we return a sample the size of a batch

    def add(self, experience: Experience) -> None:
        """Record the experience.

        Note: because we use a limited size deque old experiences drop out from the set we sample.  This is currently
        the only device for using only recent experiences.
        """
        self.memory.append(experience)

    def sample(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """Get a sample of the recorded experiences.

        :raise ValueError: when trying to get a sample that is larger than the number of recorded experiences.
        """
        sample = random.sample(self.memory, k=self.sample_size)

        def to_float_tensor(dat):
            # noinspection PyUnresolvedReferences
            return torch.from_numpy(np.vstack(dat)).float().to(device)

        states_tensor = to_float_tensor([e.state for e in sample])
        actions_tensor = to_float_tensor([e.action for e in sample])
        rewards_tensor = to_float_tensor([e.reward for e in sample])
        next_states_tensor = to_float_tensor([e.next_state for e in sample])
        dones_tensor = to_float_tensor(np.array([e.done for e in sample], dtype=np.uint8))

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def choice(self):
        """Get one experience from the memory, leave as numpy objects"""
        return random.choice(self.memory)

    def __len__(self):
        return len(self.memory)
