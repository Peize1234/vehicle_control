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
from PolicyModels import PPOPlus, Fuser
from env import SimEnv
from ErrorEncoder import ErrorEncoder
from ClassicalController import ClassicalController
from MotionModel import BicycleModel, state_space_dim, state_space_aug_dim
from utils import state_diff, MultiListContainer
from LinearTrackingAdaptiveController import LinearTrackingAdaptiveController
from train_stage2 import construct_sequence_input


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "AutoDrive-v2.6-调参"
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
    frame_delay = 0.001             # if required; add delay b/w frames

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
    stage = 2
    env = SimEnv(trace_path, num_vehicles=num_vehicles, stage=stage)

    # state_space space dimension
    state_dim = (env.observation_space.shape[0] - 2) + env.trace_points.shape[1] * 2

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    # error encoder parameters
    error_encoder_input_dim = env.observation_space.shape[0] + 2
    path_state_out_dim = 10
    error_encoder_out_dim = 1
    error_encoder_seq_len = 10
    error_encoder_num_heads = 8
    error_encoder_width = 1
    error_encoder_drop_prob = 0.1

    error_encoder = ErrorEncoder(error_encoder_input_dim, path_state_out_dim, error_encoder_out_dim,
                                 error_encoder_seq_len, error_encoder_num_heads, error_encoder_width,
                                 error_encoder_drop_prob)

    # classical controller
    # base_control_method = "front_wheel_feedback"
    # classical_controller = ClassicalController(trace_path, env.num_vehicles, state_space_dim)
    classical_controller = LinearTrackingAdaptiveController(trace_path, num_vehicles)

    # bicycle model
    # base_motion_model = BicycleModel(None)
    state_action_processor = SimEnv(trace_path, num_vehicles, stage, show=False)

    # fuser model parameters
    fuser_state_input_dim = state_space_dim - 2 + (env.dest_points_num + 1) * env.env_dim
    fuser_output_dim = 10
    fuser_hidden_dim = 128
    fuser_num_head = 8
    fuser_dropout = 0.1

    fuser = Fuser(fuser_state_input_dim, fuser_output_dim,
                  fuser_hidden_dim, fuser_num_head, fuser_dropout)

    # used for raw ppo policy input
    policy_input_dim = path_state_out_dim + fuser_output_dim * 2
    # policy_input_dim = 27

    ppo_agent = PPOPlus(policy_input_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        classical_controller,
                        state_action_processor,
                        error_encoder,
                        fuser,
                        action_std,
                        env.num_vehicles)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    # checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    checkpoint_path = r"C:\d\PythonCode\vehicle_control\PPO_preTrained\AutoDrive-v2.6-调参\PPO_AutoDrive-v2_best_-1362.85062.6-调参_0_0"
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    # base_control_method = "front_wheel_feedback"
    # classical_controller = ClassicalController(trace_path, num_vehicles, state_space_dim)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state_action_seq = MultiListContainer(["state", "action"], env.num_vehicles)

        # state = env.reset(np.zeros(env.num_vehicles, dtype=int))
        state = env.reset(np.arange(0, env.target_points.shape[0], env.target_points.shape[0] // env.num_vehicles)[:-1])

        u_rand = np.random.uniform(0.7, 0.8, size=max_ep_len)
        delta_est = classical_controller.init_adaptive_param()

        if render:
            plt.ion()
            # env.render()

        for t in range(1, max_ep_len + 1):

            # abs_action = classical_controller.get_action(state[:, :state_space_aug_dim],
            #                                              method=base_control_method, zero_Fxf=True).copy()
            classical_action_normalized, delta_est = classical_controller.get_action(state[:, :state_space_aug_dim],
                                                                                     done=env.done,
                                                                                     adaptive_delta=delta_est,
                                                                                     zero_Fxf=True,
                                                                                     with_adaptive=False,
                                                                                     action_normalize=True)

            state_action_seq.append("state", state, env.done)
            state_action_seq.append("action", classical_action_normalized, env.done)

            # 选择没有结束的车辆的状态和动作序列，构造为长度为 error_encoder_seq_len + 1 的时序输入 (取最后n个状态和动作)
            input_state = construct_sequence_input(state_action_seq, error_encoder_seq_len + 1, env.done)

            # select action with policy
            # print(state.shape)
            # if np.any(input_state > 4000) or np.any(input_state[:, :, :2] < -3):
            #     print("input_state > 4000")
            relative_action_normalized = ppo_agent.select_action(input_state, env.done)

            print("classical_action_normalized: ", classical_action_normalized)
            print("RL_action_normalized: ", relative_action_normalized)

            total_action = np.clip(classical_action_normalized + relative_action_normalized, -1, 1)
            state_action_seq.set_item("action", total_action, index=-1, done=env.done)

            state, reward, done = env.step(total_action, normalized=True, test=True)

            if t % 50 == 0:
                print(t, state)
            if len(reward) != 0:
                ep_reward += np.mean(reward)

            if render:
                env.render(show_dest_points_rate=False)
                # time.sleep(frame_delay)

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
