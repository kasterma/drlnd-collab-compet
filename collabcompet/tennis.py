# Tennis
#
# This class enables all interaction with the unity tennis environment.


import logging.config
from typing import List

import numpy as np
import yaml
from unityagents import UnityEnvironment

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("environment")


class StepResult:
    """Wrapper object for the information extracted from the unity environment"""

    def __init__(self, done: bool, rewards: List, next_state: np.ndarray):
        # Note: we check some types here, not very pythonic, but mostly to check understanding of the environment
        # This understanding will help avoid errors later on when using this data.
        assert type(done[0]) == bool and type(done) == list
        assert type(rewards) == list and len(rewards) == 2
        assert next_state.shape == (2, 24)
        self.done = done
        self.rewards = rewards
        self.next_state = next_state


class Tennis:
    def __init__(self):
        self.env = UnityEnvironment(file_name="files/Tennis.app")
        log.debug("Tennis environment set up")

    def reset(self, train_mode=True) -> np.ndarray:
        """Reset the environment

        :param train_mode boolean to indicate whether to start environment in train mode
        :return the initial state
        """
        env_info = self.env.reset(train_mode=train_mode)["TennisBrain"]
        log.debug("Tennis environment reset with train_mode=%s", train_mode)
        return env_info.vector_observations

    def step(self, action: np.ndarray) -> StepResult:
        """Take the action in the environment and collect the relevant information to return from the environment"""
        log.debug("taking step")
        assert action.shape == (2, 2)
        action_clipped = np.clip(action, -1, 1)
        env_info = self.env.step(action_clipped)["TennisBrain"]
        # check the assumption that the end of an interaction is determined by a number of steps, hence all are done
        # at the same time
        if np.any(env_info.local_done):
            assert np.all(env_info.local_done)
        return StepResult(done=env_info.local_done,
                          rewards=env_info.rewards,
                          next_state=env_info.vector_observations)
