# 对应于 train_stage2.py 的测试部分

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
from MotionModel import state_space_dim, state_space_aug_dim
from ClassicalController import ClassicalController


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "AutoDrive-v2"
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
    num_vehicles = 5
    trace_path = "trace/sweep.npy"
    env = SimEnv(trace_path, num_vehicles=num_vehicles, stage=2)

    # state_space space dimension
    state_dim = (env.observation_space.shape[0] - 2) + env.trace_points.shape[1] * 2

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(27, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, action_std, num_vehicles)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    base_control_method = "front_wheel_feedback"
    classical_controller = ClassicalController(trace_path, num_vehicles, state_space_dim)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset(np.arange(0, env.target_points.shape[0], env.target_points.shape[0] // env.num_vehicles)[:-1])

        if render:
            plt.ion()
            # env.render(show_dest_points_rate=False)
        for t in range(1, max_ep_len+1):
            abs_action = classical_controller.get_action(state[:, :state_space_aug_dim],
                                                         method=base_control_method, zero_Fxf=True).copy()
            abs_action /= env.action_space.high  # normalize action to [-1, 1]
            state = np.hstack((state[:, 2:state_space_dim], state[:, state_space_aug_dim:], abs_action[:, 0:1]))

            action = ppo_agent.select_action(state, env.done)
            action = np.clip(action + abs_action, -1, 1)

            # print(action)
            _, _, _, state, reward, done = env.step(action, normalized=True)

            if t % 50 == 0:
                print(t, state)
            if len(reward) != 0:
                ep_reward += np.mean(reward)

            if render:
                env.render(show_dest_points_rate=False)
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
