# 使用传统控制器和强化学习控制器共同控制车辆运行的强化学习训练代码

import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool

from PPO import PPO
from PolicyModels import PPOPlus, Fuser
from env import SimEnv
from ErrorEncoder import ErrorEncoder
from ClassicalController import ClassicalController
from MotionModel import BicycleModel, state_space_dim, state_space_aug_dim
from utils import state_diff, MultiListContainer
from Trace import Trace
from LinearTrackingAdaptiveController import LinearTrackingAdaptiveController
from copy import deepcopy


u_ref = 0.7

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "AutoDrive-v2.5"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 4000  # max timesteps in one episode
    max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 2 * 0.01  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # env = gym.make(env_name)
    trace_path = "trace/sweep.npy"
    num_vehicles = 5  # number of vehicles in the environment
    stage = 2
    env = SimEnv(trace_path, num_vehicles=num_vehicles, stage=stage)

    # state_space space dimension
    # state_dim = env.observation_space.shape[0] + 2 * 5

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    # run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    ppo_checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    error_encoder_checkpoint_name = "encoder_state_dict_width1.pth"
    print("save checkpoint path : " + ppo_checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state_space space dimension : ", state_space_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

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
    state_action_processor = SimEnv(trace_path, num_vehicles, stage)

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
    # ppo_agent = PPO(policy_input_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                 has_continuous_action_space, action_std, num_vehicles)

    if os.path.exists(ppo_checkpoint_path):
        # ppo_agent.load(ppo_checkpoint_path)
        print("loaded pre-trained ppo weights!")

    if os.path.exists(error_encoder_checkpoint_name):
        error_encoder.load_encoder_state_dict(error_encoder_checkpoint_name)
        error_encoder.freeze_encoder()
        print("loaded pre-trained error encoder weights!")

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    state_action_seq_max_len = max_ep_len
    assert state_action_seq_max_len >= error_encoder_seq_len + 1, \
        "state_action_seq_max_len should be greater than or equal to error_encoder_seq_len + 1"

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset(np.zeros(env.num_vehicles, dtype=int))
        # state = env.reset(np.arange(0, env.target_points.shape[0], env.target_points.shape[0] // env.num_vehicles)[:-1])

        current_ep_reward = np.zeros(env.num_vehicles)
        state_action_seq = MultiListContainer(["state", "action"], env.num_vehicles)

        u_rand = np.random.uniform(0.7, 0.8, size=max_ep_len)
        for t in range(1, max_ep_len + 1):

            # abs_action = classical_controller.get_action(state[:, :state_space_aug_dim],
            #                                              method=base_control_method, zero_Fxf=True).copy()
            classical_action_normalized = classical_controller.get_action(state[:, :state_space_aug_dim],
                                                                          done=env.done,
                                                                          zero_Fxf=True,
                                                                          with_adaptive=True,
                                                                          action_normalize=True).copy()

            state_action_seq.append("state", state, env.done)
            state_action_seq.append("action", classical_action_normalized, env.done)

            # 选择没有结束的车辆的状态和动作序列，构造为长度为 error_encoder_seq_len + 1 的时序输入 (取最后n个状态和动作)
            input_state = construct_sequence_input(state_action_seq, error_encoder_seq_len + 1, env.done)

            # select action with policy
            # print(state.shape)
            if np.any(input_state > 4000) or np.any(input_state[:, :, :2] < -3):
                print("input_state > 4000")
            relative_action_normalized = ppo_agent.select_action(input_state, env.done)

            total_action = np.clip(classical_action_normalized + relative_action_normalized, -1, 1)
            state_action_seq.set_item("action", total_action, index=-1, done=env.done)

            (state_add_to_buffer, reward_add_to_buffer, done_add_to_buffer, state, reward, output_done) = env.step(
                total_action, normalized=True)

            # saving reward and is_terminals
            j = 0
            for i in range(len(output_done)):
                if not output_done[i]:
                    ppo_agent.buffer.rewards[i].append(reward_add_to_buffer[j])
                    ppo_agent.buffer.is_terminals[i].append(done_add_to_buffer[j])
                    ppo_agent.buffer.u[i].append(torch.tensor([u_rand[t-1]]))
                    j += 1

            time_step += 1
            current_ep_reward[~output_done] += reward_add_to_buffer

            # if time_step % 500 == 0:
            #     print("time_step: ", time_step)

            # update PPO agent
            if time_step % update_timestep == 0:
                # print("updating ppo agent...")
                ppo_agent.update()
                zero_start_data_num = 0
                middle_start_data_num = 0

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = np.round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = np.round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + ppo_checkpoint_path)
                ppo_agent.save(ppo_checkpoint_path)
                error_encoder.save_all_state_dict()
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if np.all(env.done):
                break
        # print("----------"*10)
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


def construct_sequence_input(state_action_seq_container: MultiListContainer, seq_len: int, done: np.ndarray) -> np.ndarray:
    """
    construct sequence input for ppo agent (if seq_len > len(state_seq) 则使用第一个元素对之前的空缺进行补全)
    :param state_action_seq_container: list of states and actions dict("state": [[(state_dim, ), ...], ...],
                                                                       "action": [[(action_dim, ), ...], ...])
    :param seq_len: used sequence length
    :param done: done flag of each vehicle (batch_size(vehicle_num),)
    :return: sequence input for ppo agent. shape: (batch_size(num_not_done), seq_len, state_dim + action_dim)
    """
    state_seq_arr = np.stack(np.array(state_action_seq_container.get("state"),
                             dtype=object)[~done]).astype(np.float32)[:, -seq_len:, :]
    action_seq_arr = np.stack(np.array(state_action_seq_container.get("action"),
                              dtype=object)[~done]).astype(np.float32)[:, -seq_len:, :]
    assert state_seq_arr.shape[:2] == action_seq_arr.shape[:2]

    if state_seq_arr.shape[1] < seq_len:
        state_seq_arr = np.concatenate((np.repeat(state_seq_arr[:, 0:1, :], seq_len - state_seq_arr.shape[1], axis=1),
                                        state_seq_arr), axis=1)
        action_seq_arr = np.concatenate((np.repeat(action_seq_arr[:, 0:1, :], seq_len - action_seq_arr.shape[1], axis=1),
                                         action_seq_arr), axis=1)
    input_state = np.concatenate((state_seq_arr, action_seq_arr), axis=-1)
    # TODO: using dict data structure is better
    # input_state = {"state_seq": state_seq_arr, "action_seq": action_seq_arr}
    return input_state


def update_sequence(state_seq: np.array, action_seq: list, state, action, seq_max_len):
    """
    update state_seq and action_seq
    :param state_seq: list of states [(batch_size, state_dim), ...]
    :param action_seq: list of actions [(batch_size, action_dim), ...]
    :param state: new state (batch_size, state_dim)
    :param action: new action (batch_size, action_dim)
    :return: updated state_seq and action_seq (list)
    """
    state_seq.append(state)
    action_seq.append(action)
    if len(state_seq) > seq_max_len:
        state_seq.pop(0)
        action_seq.pop(0)
    return state_seq, action_seq


if __name__ == '__main__':
    train()







