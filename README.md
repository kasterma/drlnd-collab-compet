# Project: Collaboration and Competition

Third project of the udacity nano-degree on reinforcement learning.

To setup run

    make setup
    
this will install the virtual env, and download e.g. the unity environment.

Then we can run the training by running

    python train.py train --run_id=<ID> --maddpg
    
where <ID> is a numeric run identifier.  To then run the trained models without noise in the environment run

    python train.py evaluate --run_id=<ID>
    
# Environment

![Tennis environment](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it
receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward
of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each
agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or
away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100
consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each
  agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Training results

The saved weights and scores of the successful training run are in the `data` directory.

# Code

All the code other than the driver `train.py` and ad hoc plot code `evaluate.py` is in the collabcompet directory.  The
logging configuration is in `logging.yaml` and the training and network configuration is in `config.yaml`.
