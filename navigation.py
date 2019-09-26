#!/usr/bin/env python3
import os
import random
import torch
import argparse
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time

from unityagents import UnityEnvironment

from dqn_agent import Agent


def train(agent, env, dst, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Train Deep Q-Learning.
    
    Params
    ======
        agent (Agent): agent
        env (unityagents): Unity environment
        dst (str): destination path for checkpoint
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                                                  # list containing scores from each episode
    scores_window = deque(maxlen=100)                            # last 100 scores
    avg100_scores = []                                           # mean window score
    eps = eps_start                                              # initialize epsilon
    start = time()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]        # reset the environment
        state = env_info.vector_observations[0]                  # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                       # dqn_agent.py
            env_info = env.step(action)[brain_name]              # send the action to the environment
            next_state = env_info.vector_observations[0]         # get the next state
            reward = env_info.rewards[0]                         # get the reward
            done = env_info.local_done[0]                        # see if episode has finished
            agent.step(state, action, reward, next_state, done)  # dqn_agent.py
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)                              # save most recent score
        scores.append(score)                                     # save most recent score
        avg100_score = np.mean(scores_window)                    # mean window score
        avg100_scores.append(avg100_score)                       # save men window score
        eps = max(eps_end, eps_decay*eps)                        # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if avg100_score>=13.0:  # taget: an average score of +13 over 100 consecutive episodes.
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, avg100_score))
            torch.save(agent.qnetwork_local.state_dict(), dst)
            print('\nTime: {:.2f} mins'.format((time()-start)/60))
            break
    return scores, avg100_scores

def play(agent, env, num_eps, checkpoint):
    """Play Deep Q-Learning model.
    
    Params
    ======
        agent (Agent): agent
        env (unityagents): Unity environment
        num_eps (int): maximum number of playing episodes
        checkpoint (str): model path for loading
    """
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint))
    print('{} loaded'.format(checkpoint))

    for i in range(1, num_eps+1):
        score = 0 
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:
                break
        print("Episode #{} Score: {}".format(i, score))

def plot_and_save(scores, filename):
    """Plot and save scores and graph.
    
    Params
    ======
        scores (Tuple[float]): tuple of scores and average scores over 100 episodes
        filename (str): name of file
    """
    # save scores
    file_pth = os.path.join('report', '{}.npy'.format(filename))
    np.save(file_pth, np.asarray(scores))
    # plot the scores
    scores, avg100_scores = scores
    img_pth = os.path.join('report', '{}.png'.format(filename))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(avg100_scores)), avg100_scores)
    plt.title('{} DQN'.format(filename))
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(['score', 'avg_100'])
    plt.savefig(img_pth)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play_eps", default=0, action="store", type=int, help="Train if 0 else play episodes, default 0")
    parser.add_argument("--env_file", default="Banana.app", action="store", help="Unity environment binary file, default Banana.app")
    parser.add_argument("--dueling", default=False, action="store_true", help="Enable dueling DQN")
    parser.add_argument("--double", default=False, action="store_true", help="Enable double DQN")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load Unity environment
    env = UnityEnvironment(file_name=args.env_file)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    # Initialize Agent
    print('Double: {}, Dueling: {}'.format(args.double, args.dueling))
    agent = Agent(state_size, action_size, seed=0, double=args.double, dueling=args.dueling)

    checkpoint = 'model/basic.pth'
    if args.double:
        checkpoint = 'model/double.pth'
    if args.dueling:
        checkpoint = 'model/dueling.pth'
    if args.double and args.dueling:
        checkpoint = 'model/double_dueling.pth'
    print('Checkpoint: {}'.format(checkpoint))
    
    # Train or play
    if args.play_eps == 0:
        scores = train(agent, env, checkpoint)
        basename = os.path.basename(checkpoint)
        filename = os.path.splitext(basename)[0]
        plot_and_save(scores, filename)
    else:
        play(agent, env, args.play_eps, checkpoint)

    env.close()
