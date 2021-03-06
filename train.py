# Training driver for the DRLND collaboration competition project
#
# This code is the primary interface to start and evaluate the training for this project.
import logging
from collections import deque

import click
import numpy as np
from datetime import datetime

import torch

from collabcompet import *
from collabcompet.agents import MADDPG
from collabcompet.config import config
from collabcompet.tbWrapper import TBWrapper
from collabcompet.orm import load_config_from_db, session, Model, CriticInput, CriticValue, save_scalar

log = logging.getLogger("agent")

DATA_DIR = config['data_dir']


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
@click.option('--number_episodes', default=10_000, help='Number of episodes to train for.')
@click.option('--print_every', default=1, help='Print current score every this many episodes')
@click.option('--continue_run/--no-continue_run', default=False,
              help='Indicator for whether this is a continue of earlier run')
@click.option('--graphics/--no-graphics', default=True)
@click.option('--continue_run_id', default=1, help="The id of the run to continue")
@click.option('--save_models_every', default=50, help='Save the models trained every this many episodes')
@click.option('--steps_after', default=100, help="Number of steps to take after 'trained'.")
def train_run(number_episodes: int, print_every: int, continue_run: bool, graphics: bool, continue_run_id: int,
              save_models_every: int, steps_after: int, scores_window: int = 100):
    """Perform a training run

    :param graphics:
    :param steps_after:
    :param save_models_every: sve the models being trained every this many episodes
    :param maddpg: flag for use of maddpg agent versus independent agents
    :param continue_run: flag indicating this run should be a continuation of an earlier run
    :param continue_run_id: the run id for the run we are continuing
    :param scores_window: length of window to average over to check if goal has been reached
    :param number_episodes the number of episodes to run through
    :param print_every give an update on progress after this many episodes
    """
    start_run(note="Training run")
    run_id = current_runid()
    log.info("Run with id %s", run_id)
    env = Tennis(no_graphics=not graphics)

    tb = TBWrapper('./logs/logs-run_id_{}-{}'.format(run_id, datetime.now()))

    agent: MADDPG = MADDPG(replay_memory_size=config['replay_memory_size'],
                           state_size=config['state_size'],
                           action_size=config['action_size'],
                           actor_count=config['actor_count'],
                           run_id=run_id)

    if continue_run:
        assert continue_run_id is not None
        log.info("Continuing run")
        max_episode_idx = np.max([m.episode_idx for m in session.query(Model).filter_by(run_id=continue_run_id).all()])
        log.info(f"model episode id {max_episode_idx}")
        agent.load(continue_run_id, int(max_episode_idx))
    else:
        log.info("Starting new run")
    state = env.reset(train_mode=True)
    scores = []
    critic_inputs = []
    scores_deque = deque(maxlen=scores_window)
    max_mean_achieved = -1
    trained_episode_idx = -1
    try:
        for episode_idx in range(number_episodes):
            # save models at the start of the run
            if episode_idx % save_models_every == 0:
                agent.save(episode_idx, f"eps_{episode_idx}")
            env.reset(train_mode=True)
            agent.reset()
            score = np.zeros(2)
            score_list = []
            while True:
                action = agent.get_action(state, add_noise=episode_idx < 3000)
                assert action.shape == (2, 2)
                step_result = env.step(action)

                experience = Experience(state, action, step_result.rewards, step_result.next_state, step_result.done,
                                        joint=True)
                agent.record_experience(experience)
                tb.add_scalar("actor_a", agent.loss[0])
                save_scalar(episode_idx, "actor_a", agent.loss[0])
                tb.add_scalar("actor_b", agent.loss[1])
                save_scalar(episode_idx, "actor_b", agent.loss[1])
                tb.add_scalar("critic", agent.loss[2])
                save_scalar(episode_idx, "critic", agent.loss[2])
                # print(step_result.rewards)
                score += step_result.rewards
                score_list.append(step_result.rewards)
                if np.any(step_result.done):
                    break
                state = step_result.next_state
            log.debug(f"Score list length {len(score_list)}")
            log.debug(f"Max score {np.max(np.sum(score_list, axis=0))}")
            log.debug(f"Discounted max score {np.max(([0.99**i for i in range(np.array(score_list).shape[0])] * np.array(score_list).transpose()).transpose())}")
            assert score.shape == (2,)
            episode_score = np.max(score)
            save_score(episode_idx, episode_score)
            scores.append(episode_score)
            scores_deque.append(episode_score)
            mean_achieved_score = np.mean(scores_deque)
            max_mean_achieved = max(mean_achieved_score, max_mean_achieved)
            tb.add_scalar("score".format(run_id), episode_score)
            save_scalar(episode_idx, "score", episode_score)
            tb.add_scalar("mean-scores".format(run_id), mean_achieved_score)
            save_scalar(episode_idx, "mean-scores", mean_achieved_score)
            if episode_idx % print_every == 0:
                log.info("Mean achieved score %f (max %f)  ---  %d/%d (%f)",
                         mean_achieved_score, max_mean_achieved, episode_idx, number_episodes, episode_score)

            if trained_episode_idx < 0 and mean_achieved_score > 0.5:
                log.info("train success")
                trained_episode_idx = episode_idx

            if trained_episode_idx > 0 and episode_idx > trained_episode_idx + steps_after:
                break

            if episode_idx % 10 == 0:
                # add a new state action pair to track; we add them over time since there will be new combinations
                # that appear later as the agents learn to play better.  The speed with which to add should be
                # tweaked to get good data while not slowing learning too much
                new_critic_input: Experience = agent.experiences.choice()
                new_critic_state = torch.from_numpy(new_critic_input.state).float().unsqueeze(0)
                new_critic_actions = torch.from_numpy(new_critic_input.action).float().unsqueeze(0)
                ci = CriticInput(state=new_critic_state, actions=new_critic_actions)
                orm.session.add(ci)
                orm.session.commit()
                critic_inputs.append(ci)

            for ci in critic_inputs:
                critic_value = agent.critic_local.forward(ci.state, ci.actions)
                log.debug(critic_value)
                cv = CriticValue(run_id=run_id, episode_idx=episode_idx, input_id=ci.id,
                                 value1=critic_value[0,0], value2=critic_value[0,1])
                orm.session.add(cv)
                orm.session.commit()
    except KeyboardInterrupt:
        log.info("Stopped early by keyboard interrupt")
    log.info(f"Saving final models under id {run_id} and label {episode_idx + 1}")
    agent.save(episode_idx + 1, "final train value")


@click.command()
@click.option('--number_episodes', default=100, help='Number of episodes to evaluate for.')
@click.option('--print_every', default=1, help='Print current score every this many episodes')
@click.option('--run_id', help='Run id for the model to evaluate.', type=int)
@click.option('--label', default="", help="additional label under which the model was stored in the db")
def evaluation_run(number_episodes: int, print_every: int, run_id: int, label: str, scores_window=100):
    start_run(note=f"Evaluation run with models from run {run_id} with label {label}")
    load_config_from_db(run_id)
    log.info("Evaluate run with id %s", run_id)
    env = Tennis()
    # TODO: get these arguments from the database (runs.config column)
    agent: AgentInterface = MADDPG(replay_memory_size=config['replay_memory_size'],
                                   state_size=config['state_size'], action_size=config['action_size'],
                                   actor_count=config['actor_count'],
                                   run_id=current_runid())
    max_episode_idx = np.max([m.episode_idx for m in session.query(Model).filter_by(run_id=run_id).all()])
    agent.load(run_id=run_id, episode_idx=int(max_episode_idx))
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
            action = agent.get_action(state, add_noise=False)
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


tennis.add_command(random_test_run, name="randomrun")
tennis.add_command(train_run, name="train")
tennis.add_command(evaluation_run, name="evaluate")

if __name__ == "__main__":
    tennis()
