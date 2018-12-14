# Training driver for the DRLND continuous project
#
# This code is the primary interface to start and evaluate the training for the continous project.
import logging.config
from collections import deque

import yaml

from collabcompet import *
import numpy as np

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")


def random_test_run():
    """Take 100 randomly generated steps in the environment

    In this interaction also interact with the Agent (to test this interaction while developing it)
    """
    env = Tennis()
    agent = Agent(10000, action_size=4, actor_count=20, state_size=33)
    state = env.reset(train_mode=False)
    for step_idx in range(100):
        # noinspection PyUnresolvedReferences
        act_random = np.clip(np.random.randn(20, 4), -1, 1)
        step_result = env.step(act_random)
        for agent_idx in range(20):
            experience = Experience(state[agent_idx, :],
                                    act_random[agent_idx, :],
                                    step_result.rewards[agent_idx],
                                    step_result.next_state[agent_idx, :],
                                    step_result.done[agent_idx])
            agent.record_experience(experience)
        if np.any(step_result.done):
            break
    agent.experiences.sample()


# random_test_run()


def train_run(number_episodes: int =1000, print_every: int =1, run_id=0, scores_window=100):
    """Perfor a training run

    :param number_episodes the number of episodes to run through
    :param max_t max length of an episode
    :param print_every give an update on progress after this many episodes
    :param run_id id to use in saving models
    """
    log.info("Run with id %s", run_id)
    env = Tennis()
    agent_A = Agent(replay_memory_size=100000, state_size=24, action_size=2, actor_count=1)
    agent_B = Agent(replay_memory_size=100000, state_size=24, action_size=2, actor_count=1)
    state = env.reset(train_mode=False)
    scores = []
    scores_deque = deque(maxlen=scores_window)
    for episode_idx in range(number_episodes):
        env.reset()
        agent_A.reset()
        agent_B.reset()
        score = np.zeros(2)
        while True:
            action_A = agent_A.get_action(state[0, :])
            action_B = agent_B.get_action(state[1, :])
            step_result = env.step(np.vstack([action_A, action_B]))
            experience_A = Experience(state[0, :],
                                    action_A,
                                    step_result.rewards[0],
                                    step_result.next_state[0, :],
                                    step_result.done[0])
            agent_A.record_experience(experience_A)
            experience_B = Experience(state[1, :],
                                    action_B,
                                    step_result.rewards[1],
                                    step_result.next_state[1, :],
                                    step_result.done[1])
            agent_B.record_experience(experience_B)
            print(step_result.rewards)
            score += step_result.rewards # [agent_idx]  # TODO: score???
            print(score)
            if np.any(step_result.done):
                break
            state = step_result.next_state
        # TODO: scores max of two players
        scores.append(score/2)
        scores_deque.append(score/2)
        if episode_idx % print_every == 0:
            log.info("%d Mean score over last %d episodes %f", episode_idx, scores_window, np.mean(scores_deque))
        if np.mean(scores_deque) > 30:
            log.info("train success")
            break
    log.info("Saving models under id %s", run_id)
    agent_A.save(f"agent-A-{run_id}")
    agent_B.save(f"agent-B-{run_id}")
    log.info("Saving scores to file scores-%d.npy", run_id)
    np.save("scores-{}.npy".format(run_id), np.array(scores_deque))

train_run(run_id=0)


def test_run(number_episodes: int = 100, print_every: int = 1, run_id=0, scores_window=100):
    log.info("Run test with id %s", run_id)
    env = Tennis()
    agent = Agent(replay_memory_size=100000, state_size=33, action_size=4, actor_count=20)
    agent.load(run_id)
    state = env.reset(train_mode=True)
    scores = []
    scores_deque = deque(maxlen=scores_window)
    for episode_idx in range(number_episodes):
        env.reset(train_mode=True)
        score = 0
        ct = 0
        while True:
            ct += 1
            # noinspection PyUnresolvedReferences
            action = agent.get_action(state, add_noise=False)
            step_result = env.step(action)
            #print(step_result.rewards)
            score += np.mean(step_result.rewards)
            if np.any(step_result.done):
                break
            state = step_result.next_state
        scores.append(score)
        scores_deque.append(score)
        if episode_idx % print_every == 0:
            log.info("%d Mean score over last %d episodes %f (%d)", episode_idx, scores_window, np.mean(scores_deque), ct)

    np.save("evaluate-scores-{}.npy".format(run_id), np.array(scores_deque))


#test_run(run_id=2)
