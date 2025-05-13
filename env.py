"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from Model_Trace_Interactor import ModelTraceInteractor
import matplotlib.pyplot as plt
from MotionModel import state_space_dim, state_space_aug_dim


class SimEnv(ModelTraceInteractor, gym.Env):
    def __init__(self, trace_path: str, num_vehicles: int = 50, stage: int = 1):
        assert num_vehicles > 0, 'num_vehicles must be greater than 0'

        # zero state_space space for init params
        # 如果车辆的数量为0，则默认为轨迹点的数量减1
        super().__init__(np.zeros((num_vehicles, state_space_dim)), trace_path)

        self.num_vehicles = num_vehicles
        self.stage = stage

        self.m = 1500
        self.Iz = 2250
        self.a = 1.04
        self.b = 1.42
        self.h = 0.6
        self.g = 9.81
        self.Cxf = 2.5e3
        self.Cxr = 2.5e3
        self.Cyf = 160e3
        self.Cyr = 180e3
        self.sigma_f = 0.3
        self.sigma_r = 0.3

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 5

        self.action_space = spaces.Box(- self.action_high, self.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(- self.observation_high, self.observation_high, dtype=np.float32)

        self.fig, self.ax = plt.subplots()

        # init state_space(observation) space limit
        self.phi_limit = [-np.deg2rad(30), np.deg2rad(30)]
        # self.phi_limit = [-np.deg2rad(0), np.deg2rad(360)]
        self.Ux_limit = [0, 27]  # 100 km/h
        self.Uy_limit = [0, 5]  # 20 km/h
        self.r_limit = [-np.deg2rad(30), np.deg2rad(30)]

        # game end control params
        # 车辆中心与其到轨迹最近点的距离的最大值
        self.max_distance = 2
        # 车辆前进方向与最近点方向向量的夹角的最大值
        self.max_deviation_angle = np.deg2rad(20)
        # 车辆速度的最大值
        self.max_speed_x = self.Ux_limit[1]
        self.max_speed_y = self.Uy_limit[1]

        self.seed()
        self.init = False

        # for calculate reward
        self.dest_points_num = 10
        self.gap_distance = 1.0
        self.weight_factor = 2.0

        self.env_dim = 2

        # 运行时变量(运行时变量的形状不发生改变)
        self.state_space = np.empty((self.num_vehicles, self.observation_space.shape[0]))
        self.old_done = np.zeros(self.num_vehicles, dtype=bool)
        self.done = np.zeros(self.num_vehicles, dtype=bool)
        self.trace_points = np.empty(
            (self.num_vehicles, self.dest_points_num + 1, self.env_dim))  # first points is closed points
        self.trace_points_rates_norm = np.empty((self.num_vehicles, self.dest_points_num + 1, self.env_dim))
        self.closed_curve_idx = np.empty(self.num_vehicles, dtype=int)
        self.closed_t = np.empty(self.num_vehicles)

        # action
        self.delta = np.empty(self.num_vehicles)
        self.Fxf = np.empty(self.num_vehicles)

    @property
    def closed_points_and_rates(self):
        return self.trace_points[~self.done, 0, :], self.trace_points_rates_norm[~self.done, 0, :]

    @closed_points_and_rates.setter
    def closed_points_and_rates(self, value):
        self.trace_points[~self.done, 0, :], self.trace_points_rates_norm[~self.done, 0, :] = value

    @property
    def closed_curve_idx_and_t(self):
        return self.closed_curve_idx[~self.done], self.closed_t[~self.done]

    @closed_curve_idx_and_t.setter
    def closed_curve_idx_and_t(self, value):
        self.closed_curve_idx[~self.done], self.closed_t[~self.done] = value

    @property
    def dest_points_and_rates(self):
        return self.trace_points[~self.done, 1:, :], self.trace_points_rates_norm[~self.done, 1:, :]

    @dest_points_and_rates.setter
    def dest_points_and_rates(self, value):
        self.trace_points[~self.done, 1:, :], self.trace_points_rates_norm[~self.done, 1:, :] = value

    @property
    def actions(self):
        return self.delta, self.Fxf

    @actions.setter
    def actions(self, value):
        self.delta[~self.done], self.Fxf[~self.done] = value

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _init_env_param(self) -> None:
        """
        根据基础变量： state_space_aug 和 done 计算其余环境参数
        """
        (self.trace_points[:, 0, :], self.closed_curve_idx,
         self.closed_t) = self.get_vehicle_closed_trace_point(accuracy=1e-6)
        self.trace_points[:, 1:, :], self.trace_points_rates_norm[:, 1:, :] = self.get_n_points_rate_along_trace(
            self.closed_curve_idx,
            self.closed_t,
            gap_distance=self.gap_distance,
            requires_n_points=self.dest_points_num)
        self.trace_points_rates_norm[:, 0, :] = self.get_n_points_rate_along_trace(self.closed_curve_idx,
                                                                                   self.closed_t,
                                                                                   gap_distance=0,
                                                                                   requires_n_points=1)[0].squeeze(1)
        self.delta = np.empty(self.num_vehicles)
        self.Fxf = np.empty(self.num_vehicles)
        self.init = True

    def set_state_space(self, state_space_aug: np.ndarray,
                        done: np.ndarray = None,
                        old_done: np.ndarray = None) -> None:
        """
        设置 state_space 和 done，并基于此重新初始化环境参数
        note: done 和 old_done 必须同时给出，或者都不给出

        :param state_space_aug: 形状为 (batch_size, state_space_aug_dim)
        :param done: 形状为 (batch_size,) or None(全部不结束)，表示是否结束
        :param old_done: 形状为 (batch_size,) or None(全部不结束)，表示上一时刻是否结束
        """
        if done is None or old_done is None:
            assert done is None and old_done is None, "done and old_done must both be None or both be not None"
            done = np.zeros(state_space_aug.shape[0], dtype=bool)
            self.old_done = np.zeros(state_space_aug.shape[0], dtype=bool)
        if done is not None or old_done is not None:
            assert done is not None and old_done is not None, "done and old_done must both be None or both be not None"
            assert done.shape[0] == old_done.shape[0] == state_space_aug.shape[0], \
                "done and old_done must have the same shape as state_space_aug"

        super().set_state_space(state_space_aug, done)

        # num_vehicles may be changed after reset
        self.num_vehicles = self.state_space.shape[0]
        self._init_env_param()

    def reset(self, checkpoint: np.ndarray = None) -> np.ndarray:
        """
        Resets the environment
        :param checkpoint: 表示在哪个检查点附近生成车辆，检查点即为路径的 target_points
        """
        if checkpoint is not None:
            assert len(checkpoint) == self.num_vehicles, "checkpoint must be a list with length equal to num_vehicles"
        else:
            checkpoint = np.zeros(self.num_vehicles, dtype=int)

        #             phi Ux Uy r
        sub_factors = [2, 5, np.inf, np.inf]
        # x, y, phi, Ux, Uy, r
        self.state_space = self.np_random.uniform(low=np.array([- 1,
                                                                - 1,
                                                                self.phi_limit[0] / sub_factors[0],
                                                                self.Ux_limit[0] / sub_factors[1],
                                                                self.Uy_limit[0] / sub_factors[2],
                                                                self.r_limit[0] / sub_factors[3]]),
                                                  high=np.array([1,
                                                                 1,
                                                                 self.phi_limit[1] / sub_factors[0],
                                                                 self.Ux_limit[1] / sub_factors[1],
                                                                 self.Uy_limit[1] / sub_factors[2],
                                                                 self.r_limit[1] / sub_factors[3]]),
                                                  size=(self.num_vehicles, self.observation_space.shape[0]))
        self.state_space[:, :2] += self.target_points[checkpoint]
        self.set_state_space(np.hstack((self.state_space, np.zeros((self.num_vehicles,
                                                                    state_space_aug_dim - state_space_dim)))))
        self.done = np.zeros(self.num_vehicles, dtype=bool)

        self._init_env_param()

        return self.format_state(~self.done)

    def step(self, action: np.ndarray, normalized: bool = False, return_raw_state: bool = False) -> tuple:
        """
        Performs a single step of the environment
        :param action: (delta, Fxf) shape=(num_not_done, 2)
        :param normalized: whether the action is normalized or not
        :param return_raw_state: whether to return the raw state_space or not
        :return: state
        """
        # 防止 action *= np.array([self.action_space.high]) 修改外部传入的 action
        action = action.copy()

        if normalized:
            assert np.all(action >= -1) and np.all(action <= 1), ("%r (%s) invalid" % (action, type(action)))
            action *= np.array([self.action_space.high])
        else:
            assert np.all(action >= self.action_space.low) and np.all(action <= self.action_space.high), (
                    "%r (%s) invalid" % (action, type(action)))
        self.step_once(*action.T)

        self.old_done = self.done.copy()
        self.update_param(action)

        reward = np.zeros(self.num_vehicles)
        reward[~self.done] = self.calculate_reward()
        reward[~self.old_done & self.done] = -5000
        # reward = self.calculate_reward()

        # 更新后没有结束的车辆正常输出，同时上一时刻没有结束但是现在结束的车辆也要输出，用于训练过程中 reward 计算
        # output_select = ~self.done | (~self.old_done & self.done)
        # 等价于：(不完全等价，但是 old_done 为 True 同时 done 为 False 的情况不存在，所以在此场景中等价)
        output_select = ~ self.old_done

        # 检查正常结束的车辆，必须要放在这里，不影响 output_select，但是影响 done(用于下一次计算的状态)
        self.check_normal_done()

        state_add_to_buffer = self.format_state(output_select)
        reward_add_to_buffer = reward[output_select]
        done_add_to_buffer = self.done[output_select]

        state_next = self.format_state(~self.done)
        reward_next = reward[~self.done]
        done_next = self.done[~self.done]

        if return_raw_state:
            return self.state_space_aug

        if self.stage == 1:
            return state_add_to_buffer, reward_add_to_buffer, done_add_to_buffer, state_next, reward_next, done_next
        elif self.stage == 2:
            return state_add_to_buffer, reward_add_to_buffer, done_add_to_buffer, state_next, reward_next, ~output_select
        else:
            raise ValueError("Invalid stage")

    def format_state(self, idx):
        if np.all(~idx):
            return []

        done_copy = self.done.copy()
        self.done = ~idx
        trace_points_trans, _ = self.transfer_points_rates_to_vehicle_coordinate(self.trace_points[idx, :, :2],
                                                                                 None, "center")
        trace_points_trans_diff = trace_points_trans[:, 1:, :] - trace_points_trans[:, :-1, :]
        self.done = done_copy

        # TODO: using dict data structure is better
        if self.stage == 1:
            return np.concatenate(
                (
                    self.state_space[idx, 2:] / np.array([[self.phi_limit[1], self.Ux_limit[1],
                                                           self.Uy_limit[1], self.r_limit[1]]]),
                    trace_points_trans[:, 0, :],  # closed point in trace in vehicle coordinate
                    trace_points_trans_diff.reshape(trace_points_trans_diff.shape[0], -1)
                ),
                axis=1
            )
        elif self.stage == 2:
            state_space_aug_idx = self.state_space_aug[idx].copy()
            state_space_aug_idx[:, 2:state_space_dim] /= np.array([[self.phi_limit[1], self.Ux_limit[1],
                                                                    self.Uy_limit[1], self.r_limit[1]]])
            return np.concatenate(
                (
                    state_space_aug_idx,
                    trace_points_trans[:, 0, :],  # closed point in trace in vehicle coordinate
                    trace_points_trans_diff.reshape(trace_points_trans_diff.shape[0], -1)
                ),
                axis=1
            )
        else:
            raise ValueError("Invalid stage")

    def check_normal_done(self):
        """
        Checks if the vehicles are normally done and update some variables.
        """
        # 到达最后一条线段(额外插入的线段)时，表示车辆走完了所有的路径，正常结束
        normal_done = self.closed_curve_idx >= self.point_type_idx.shape[0] - 1
        self.done[normal_done] = True

    def update_param(self, action):
        # (num_not_done, ...)
        closed_points, closed_curve_idx, closed_t = self.get_vehicle_closed_trace_point(accuracy=1e-6)
        not_done_idx = np.argwhere(~self.done)

        # 检查车辆是否超过设置的偏离轨迹最大值
        done_idx_distance = np.linalg.norm(closed_points - self.state_space[~self.done, :2],
                                           axis=1) > self.max_distance

        # (num_not_done, ...)
        _, closed_points_rate = self.get_n_points_rate_along_trace(closed_curve_idx,
                                                                   closed_t,
                                                                   gap_distance=0, requires_n_points=1)
        closed_points_rates_norm = closed_points_rate.squeeze(1)
        vehicle_rates_norm = self.forward_norm_vector
        diff_angle = np.arccos(np.sum(closed_points_rates_norm * vehicle_rates_norm, axis=1))

        # 检查车辆与最近轨迹点的夹角是否超过设置的最大值
        done_idx_angle = diff_angle > self.max_deviation_angle

        # 检查车辆速度是否超过设置的最大值 或 行驶速度是否小于0
        done_idx_speed = ((self.state_space[~self.done, 3] > self.max_speed_x) |
                          (self.state_space[~self.done, 4] > self.max_speed_y) |
                          (self.state_space[~self.done, 3] < 0))

        # 更新车辆的状态(是否结束)
        done_idx = done_idx_distance | done_idx_speed
        self.done[not_done_idx[done_idx]] = True

        # 更新其他参数
        self.closed_points_and_rates = closed_points[~done_idx], closed_points_rates_norm[~done_idx]
        self.closed_curve_idx_and_t = closed_curve_idx[~done_idx], closed_t[~done_idx]
        self.dest_points_and_rates = self.get_n_points_rate_along_trace(self.closed_curve_idx[~self.done],
                                                                        self.closed_t[~self.done],
                                                                        gap_distance=self.gap_distance,
                                                                        requires_n_points=self.dest_points_num)
        self.actions = action[~done_idx, 0], action[~done_idx, 1]

    def calculate_reward(self):
        dest_points, dest_points_rate = self.dest_points_and_rates
        closed_points, closed_points_rate = self.closed_points_and_rates

        # (num_not_done, num_dest_points + 1(closed_point), 2)
        dest_points = np.concatenate((closed_points[:, None, :], dest_points), axis=1)
        dest_points_norm_rate = np.concatenate((closed_points_rate[:, None, :], dest_points_rate), axis=1)

        # (num_not_done, 2)
        vehicle_center = self.vehicle_center
        vehicle_norm_rate = self.forward_norm_vector

        # 靠近的点给比较大的权重，越远的点权重越小
        weight = self.get_weight_factor()

        # 每辆车距离 标定点 的距离、方向夹角
        distances = np.linalg.norm(dest_points - vehicle_center[:, None, :], axis=2)  # 越小越好
        angle_diff = 1 + np.sum(dest_points_norm_rate * vehicle_norm_rate[:, None, :], axis=2)  # (-1, 1) -> (0, 2) 越小越好

        # # 每辆车距离 标定点 的最大距离、方向夹角
        # max_distance = np.array([self.max_distance + i * self.gap_distance for i in range(self.dest_points_num + 1)])
        # max_angle_diff = np.ones(self.dest_points_num + 1) * 2
        #
        # # 归一化
        # distances_norm = 1 - distances / max_distance
        # angles_norm = angle_diff / max_angle_diff
        #
        # # 乘权重因子, / (self.dest_points_num + 1) 的目的是使得奖励值在 0-1 之间，奖励值越大越好
        # reward = (np.sum((distances_norm + angles_norm) * weight / (self.dest_points_num + 1), axis=1) +
        #           self.state_space[~self.done, 3] / self.max_speed_x)

        reward = np.where(distances[:, 0] < self.max_distance / 10, self.state_space[~self.done, 3], 0)

        return reward

    def get_weight_factor(self):
        x = np.linspace(0, 1, self.dest_points_num + 1, endpoint=True)[::-1]
        return self.softmax(x ** self.weight_factor)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

    def render(self, show_trace=True, show_vehicle=True, show_connect_lines=True,
               show_dest_points=True, show_dest_points_rate=True, vehicle_view=False, quiver_scale=1):
        assert self.init, "closed_points is not None, please reset the environment"

        if show_trace:
            self.show_trace(show_target_points=False, show_support_points=False, show=False)
        if show_vehicle:
            self.show_vehicle(delta=self.delta)
        if show_connect_lines:
            self.show_connect_lines(self.closed_points_and_rates[0], show=False)
        if show_dest_points:
            self.show_n_points(self.trace_points, show=False)
        if show_dest_points_rate:
            self.show_n_rate(self.trace_points, self.trace_points_rates_norm, quiver_scale=quiver_scale)
        if vehicle_view:
            self.ax.set_xlim(self.state_space[0, 0] - self.L, self.state_space[0, 0] + self.L * 2)
            self.ax.set_ylim(self.state_space[0, 1] - self.L, self.state_space[0, 1] + self.L * 2)

        plt.pause(0.01)
        plt.cla()

    def close(self):
        pass


if __name__ == '__main__':

    random_seed = 5
    np.random.seed(random_seed)

    env = SimEnv("trace/sweep.npy", num_vehicles=1)
    env.seed(random_seed)
    env.reset()
    show_animation = True

    max_step = 10000
    random_action = np.random.uniform(low=env.action_space.low, high=env.action_space.high,
                                      size=(max_step, env.num_vehicles, env.action_space.shape[0]))

    new_fig, new_ax = plt.subplots()
    if show_animation:
        plt.ion()
        env.render()
        plt.pause(0.001)
        plt.show()
    i = 0
    while True:
        print(i)
        _, _, _, state, rewards, dones = env.step(random_action[i, ~env.done])
        print(state)
        print(rewards)

        # debug
        new_ax.plot(state[:, 4::2], state[:, 5::2], "o")
        # plt.pause(0.001)
        # new_fig.clf()

        if show_animation:
            env.render()

        i += 1
        if i >= max_step:
            break

        if env.done.all():
            break

    if show_animation:
        plt.ioff()
