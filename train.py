# Training driver for the DRLND continuous project
#
# This code is the primary interface to start and evaluate the training for the continous project.
import logging.config
from collections import deque

import yaml
import click
import os.path

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

@click.command()
@click.option('--number_episodes', default=100, help='Number of episodes to train for.')
@click.option('--print_every', default=1, help='Print current score every this many episodes')
@click.option('--run_id', help='Run id for this run.', type=int)
@click.option('--continue_run', default=False, help='Indicator for whether this is a continue of earlier run')
def train_run(number_episodes: int, print_every: int, run_id: int, continue_run: bool, scores_window: int=100):
    """Perfor a training run

    :param continue_run:
    :param scores_window:
    :param number_episodes the number of episodes to run through
    :param max_t max length of an episode
    :param print_every give an update on progress after this many episodes
    :param run_id id to use in saving models
    """
    log.info("Run with id %s", run_id)
    env = Tennis()
    agent_A = Agent(replay_memory_size=100000, state_size=24, action_size=2, actor_count=1, run_id=f"agent-A-{run_id}")
    agent_B = Agent(replay_memory_size=100000, state_size=24, action_size=2, actor_count=1, run_id=f"agent-B-{run_id}")
    if continue_run:
        log.info("Continuing run")
        assert agent_A.files_exist() and agent_B.files_exist()
        agent_A.load()
        agent_B.load()
    else:
        log.info("Starting new run")
        assert not agent_A.files_exist()
        assert not agent_B.files_exist()
    state = env.reset(train_mode=True)
    scores = []
    scores_deque = deque(maxlen=scores_window)
    try:
        for episode_idx in range(number_episodes):
            env.reset(train_mode=True)
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
                # print(step_result.rewards)
                score += step_result.rewards
                # print(score)
                if np.any(step_result.done):
                    break
                state = step_result.next_state
            assert score.shape == (2,)
            scores.append(np.max(score))
            scores_deque.append(np.max(score))
            if episode_idx % print_every == 0:
                log.info("%d Mean score over last %d episodes %f", episode_idx, scores_window, np.mean(scores_deque))
            if np.mean(scores_deque) > 30:
                log.info("train success")
                break
    except KeyboardInterrupt:
        log.info("Stopped early by keyboard interrupt")
    log.info("Saving models under id %s", run_id)
    agent_A.save()
    agent_B.save()

    def scores_filename(idx):
        """Create scores filename"""
        base = "scores-{}".format(run_id)
        idx_p = f"-{idx}" if idx != 0 else ""
        ext = ".npy"
        return base + idx_p + ext
    scores_addition = 0
    while os.path.isfile(scores_filename(scores_addition)):
        scores_addition += 1
    log.info("Saving scores to file %s", scores_filename(scores_addition))
    np.save(scores_filename(scores_addition), np.array(scores_deque))


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


if __name__ == "__main__":
    train_run()