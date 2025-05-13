
# 对应于 train_raw_dl.py 的测试代码

import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool

from PPO import PPO
from env import SimEnv
import matplotlib.pyplot as plt


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "AutoDrive-v1"
    has_continuous_action_space = True
    max_ep_len = 4000
    action_std = 0.1

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # env_name = "RoboschoolWalker2d-v1"
    # has_continuous_action_space = True
    # max_ep_len = 1000           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    # env = gym.make(env_name)
    num_vehicles = 1
    env = SimEnv("trace/sweep.npy", num_vehicles=num_vehicles)

    # state_space space dimension
    state_dim = (env.observation_space.shape[0] - 2) + env.trace_points.shape[1] * 2

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        if render:
            plt.ion()
            # env.render()
        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state, env.done)
            action = np.clip(action, -1, 1)
            # print(action)
            _, _, _, state, reward, done = env.step(action, normalized=True)
            if t % 50 == 0:
                print(t)
                print(state)
            if len(reward) != 0:
                ep_reward += np.mean(reward)

            if render:
                env.render()
                time.sleep(frame_delay)

            if np.all(done):
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, np.round(ep_reward, 2)))
        ep_reward = 0

    env.close()
    if render:
        plt.ioff()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = np.round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
