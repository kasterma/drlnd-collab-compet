# Training driver for the DRLND collaboration competition project
#
# This code is the primary interface to start and evaluate the training for this project.
import logging.config
from collections import deque

import yaml
import click
import os.path

from collabcompet import *
import numpy as np

from collabcompet.agents import IndependentAgent, MADDPG

with open("logging.yaml") as log_conf_file:
    log_conf = yaml.load(log_conf_file)
logging.config.dictConfig(log_conf)
log = logging.getLogger("agent")

DATA_DIR = "data/"


@click.group()
def tennis():
    pass


@click.command(name="randomrun")
def random_test_run():
    """Take 1000 randomly generated steps in the environment

    In this interaction also interact with the Agent (to test this interaction while developing it)
    """
    env = Tennis()
    env.reset(train_mode=False)
    for step_idx in range(1000):
        # noinspection PyUnresolvedReferences
        act_random = np.clip(np.random.randn(2, 2), -1, 1)
        step_result = env.step(act_random)
        if np.any(step_result.done):
            env.reset(train_mode=False)


@click.command(name="train")
@click.option('--number_episodes', default=10000, help='Number of episodes to train for.')
@click.option('--print_every', default=1, help='Print current score every this many episodes')
@click.option('--continue_run', default=False, help='Indicator for whether this is a continue of earlier run')
@click.option('--maddpg/--no-maddpg', default=True)
def train_run(number_episodes: int, print_every: int, continue_run: bool, maddpg: bool, scores_window: int = 100):
    """Perform a training run

    :param maddpg: flag for use of maddpg agent versus independent agents
    :param continue_run: flag indicating this run should be a continuation of an earlier run
    :param scores_window: length of window to average over to check if goal has been reached
    :param number_episodes the number of episodes to run through
    :param print_every give an update on progress after this many episodes
    """
    start_run(note=f"Evaluation run with models from run {run_id}")
    run_id = current_runid()
    log.info("Run with id %s", run_id)
    env = Tennis()
    if maddpg:
        agent: AgentInterface = MADDPG(replay_memory_size=100000, state_size=24, action_size=2, actor_count=2,
                                       run_id=f"agent-A-{run_id}")
    else:
        agent: AgentInterface = IndependentAgent(run_id=run_id)
    if continue_run:
        log.info("Continuing run")
        agent.load()
    else:
        log.info("Starting new run")
        assert not agent.files_exist()
    state = env.reset(train_mode=True)
    scores = []
    scores_deque = deque(maxlen=scores_window)
    max_mean_achieved = -1
    try:
        for episode_idx in range(number_episodes):
            env.reset(train_mode=True)
            agent.reset()
            score = np.zeros(2)
            while True:
                action = agent.get_action(state)
                assert action.shape == (2, 2)
                step_result = env.step(action)
                experience = Experience(state, action, step_result.rewards, step_result.next_state, step_result.done,
                                        joint=True)
                agent.record_experience(experience)
                # print(step_result.rewards)
                score += step_result.rewards
                # print(score)
                if np.any(step_result.done):
                    break
                state = step_result.next_state
            assert score.shape == (2,)
            episode_score = np.max(score)
            save_score(episode_idx, episode_score)
            scores.append(episode_score)
            scores_deque.append(episode_score)
            mean_achieved_score = np.mean(scores_deque)
            max_mean_achieved = max(mean_achieved_score, max_mean_achieved)
            if episode_idx % print_every == 0:
                log.info("Mean achieved score %f (max %f)  ---  %d/%d (%f)",
                         mean_achieved_score, max_mean_achieved, episode_idx, number_episodes, episode_score)
            if mean_achieved_score > 0.5:
                log.info("train success")
                break
    except KeyboardInterrupt:
        log.info("Stopped early by keyboard interrupt")
    log.info("Saving models under id %s", run_id)
    agent.save()

    def scores_filename(idx):
        """Create scores filename"""
        base = DATA_DIR + "scores-{}".format(run_id)
        idx_p = f"-{idx}" if idx != 0 else ""
        ext = ".npy"
        return base + idx_p + ext

    scores_addition = 0
    while os.path.isfile(scores_filename(scores_addition)):
        scores_addition += 1
    log.info("Saving scores to file %s", scores_filename(scores_addition))
    np.save(scores_filename(scores_addition), np.array(scores))


@click.command()
@click.option('--number_episodes', default=100, help='Number of episodes to train for.')
@click.option('--print_every', default=1, help='Print current score every this many episodes')
@click.option('--run_id', help='Run id for this run.', type=int)
def evaluation_run(number_episodes: int, print_every: int, run_id=0, scores_window=100):
    start_run(note=f"Evaluation run with models from run {run_id}")
    log.info("Evaluate run with id %s", run_id)
    env = Tennis()
    agent: AgentInterface = MADDPG(replay_memory_size=100000, state_size=24, action_size=2, actor_count=2,
                                   run_id=f"agent-A-{run_id}")
    assert agent.files_exist()
    agent.load()
    state = env.reset(train_mode=False)
    scores = []
    scores_deque_max = deque(maxlen=scores_window)
    scores_deque_mean = deque(maxlen=scores_window)
    for episode_idx in range(number_episodes):
        env.reset(train_mode=False)
        score = np.zeros(2)
        ct = 0
        while True:
            ct += 1
            action = agent.get_action(state)
            step_result = env.step(action)
            score += step_result.rewards
            if np.any(step_result.done):
                break
            state = step_result.next_state
        scores.append(np.max(score))
        save_score(episode_idx, np.max(score))
        scores_deque_max.append(np.max(score))
        scores_deque_mean.append(np.mean(score))
        if episode_idx % print_every == 0:
            log.info("%d/%d Mean score over last %d episodes max: %f mean: %f (episode length: %d)",
                     episode_idx, number_episodes, scores_window,
                     np.mean(scores_deque_max), np.mean(scores_deque_mean),
                     ct)

    np.save(DATA_DIR + "evaluation-scores-{}.npy".format(run_id), np.array(scores_deque_max))


tennis.add_command(random_test_run, name="randomrun")
tennis.add_command(train_run, name="train")
tennis.add_command(evaluation_run, name="evaluate")

if __name__ == "__main__":
    tennis()
