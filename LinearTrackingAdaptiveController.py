from Model_Trace_Interactor import ModelTraceInteractor
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from gauss_legendre_integral_points import Ak, xk
from control import lqr
from env import SimEnv
import matplotlib.pyplot as plt
from utils import base_lengthways_control
from MotionModel import state_space_dim, state_space_aug_dim


class LinearTrackingAdaptiveController(ModelTraceInteractor):
    def __init__(self, trace_path: str, num_vehicles: int,
                 Q: np.ndarray = np.diag([1, 0, 1, 0]), R: np.ndarray = np.diag([1]), gamma=1):
        """
        Initialize the linear tracking controller
        :param trace_path: path of the trace file
        :param num_vehicles: number of vehicles in the simulation
        :param Q: state cost matrix, shape (state_dim, state_dim)
        :param R: control cost matrix, shape (control_dim, control_dim)
        :param gamma: discount factor used to control the tracking error between base controller and RL controller
        """
        super().__init__(np.zeros((num_vehicles, state_space_aug_dim)), trace_path)

        self.num_vehicles = num_vehicles

        self.error_state = None
        self.A, self.B = None, None
        self.A_d, self.B_d = None, None
        self.K = None
        self.delta_est = np.zeros((num_vehicles, 1))
        self.dt = 0.1
        self.C = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        self.Q = Q
        self.R = R
        self.gamma = gamma

    def init_adaptive_param(self):
        self.delta_est = np.zeros((self.num_vehicles, 1))
        return self.delta_est

    def set_adaptive_param(self, delta_est_before: np.ndarray):
        self.delta_est = delta_est_before

    def update_controller(self, new_state_space_aug: np.ndarray) -> None:
        """
        Update the controller matrix by new vehicle state
        """
        self.state_space = new_state_space_aug[:, :state_space_dim]
        self.state_space_aug = new_state_space_aug

        self.error_state = self.get_error_state()
        self.A, self.B = self.get_control_matrix()
        self.A_d, self.B_d = self.discretize()

        self.K = self.get_controller_lqr(self.Q, self.R)

    def get_error_state(self) -> np.ndarray:
        """
        Get the error state of the model from raw state space
        :return: error state, shape (batch_size, 4)
        """
        closed_points, opt_curve_num, opt_t = self.get_vehicle_closed_trace_point()
        target_rates = self.get_rate_by_curve_idx_and_t(opt_curve_num, opt_t)
        vehicle_center = self.state_space[:, :2]
        vehicle_forward_vector = self.forward_norm_vector

        trace2vehicle_vector = vehicle_center - closed_points
        e_abs = np.linalg.norm(trace2vehicle_vector, axis=1)
        e = np.where(np.cross(target_rates, trace2vehicle_vector) > 0, e_abs, -e_abs)

        phi_e = np.arccos(np.sum(vehicle_forward_vector * target_rates, axis=1))
        phi_e = np.where(np.cross(target_rates, vehicle_forward_vector) > 0, phi_e, -phi_e) + 1e-5  # 避免出现 nan

        zero = np.zeros(e.shape)
        return np.stack([e, zero, phi_e, zero]).T

    def get_control_matrix(self) -> tuple:
        """
        Get the continuous model of error described by matrix A and B (x_dot = Ax + Bu)
        :return: A, B, shape (batch_size, state_dim, state_dim) (batch_size, state_dim, control_dim)
        """
        vx = self.state_space[:, 3] + 1e-5
        zero = np.zeros(vx.shape)
        one = np.ones(vx.shape)

        factor1 = - (self.Cxf + self.Cxr) / (self.m * vx)
        factor2 = (self.Cxf + self.Cxr) / self.m * one
        factor3 = - (self.Cxf * self.a - self.Cxr * self.b) / (self.m * vx)
        factor4 = - (self.Cxf * self.a - self.Cxr * self.b) / (self.Iz * vx)
        factor5 = (self.Cxf * self.a - self.Cxr * self.b) / self.Iz * one
        factor6 = - (self.a ** 2 * self.Cxf + self.b ** 2 * self.Cxr) / (self.Iz * vx)

        A = np.array([[zero, one, zero, zero],
                      [zero, factor1, factor2, factor3],
                      [zero, zero, zero, one],
                      [zero, factor4, factor5, factor6]])

        B = np.array([[zero],
                      [self.Cxr / self.m * one],
                      [zero],
                      [self.a * self.Cxf / self.Iz * one]])

        return A.transpose(2, 0, 1), B.transpose(2, 0, 1)

    def discretize(self, method="ZOH") -> tuple:
        """
        Discretize the model described by matrix A and B with time step dt
        B_d is estimated by numerical integration of the integrand int_0^dt expm(A * tao) @ B dtao (7点Gauss-Legendre积分)
        :param dt: time step
        :param method: discretization method (just ZOH for now)
        :return: A_d, B_d, shape (batch_size, state_dim, state_dim) (batch_size, state_dim, control_dim)
        """
        A_d = expm(self.A * self.dt)

        factor1 = self.dt / 2
        factor2 = self.dt / 2
        in_var = factor1 * xk + factor2

        integrand_values = self._discretize_integrand_func(in_var)
        B_d = np.sum(integrand_values * Ak[None, :, None, None], axis=1) * factor1
        # print(self.A)
        # print(A_d)
        # print(self.B)
        # print(B_d)
        return A_d, B_d

    def _discretize_integrand_func(self, in_var) -> np.ndarray:
        """
        用于估计积分 int_0^dt expm(A * tao) @ B dtao 的近似值的被积函数
        :param in_var: 积分变量，shape 为 (xk.shape[0](不可省略), )
        :return: 各积分点处的积分值，shape 为 (batch_size, xk.shape[0], state_dim, control_dim)
        """
        A_tao = self.A[:, None, :, :] * in_var[None, :, None, None]
        exp_A_tao = expm(A_tao)
        expm_A_tao_B = exp_A_tao @ self.B[:, None, :, :]

        return expm_A_tao_B

    def get_controller_lqr(self, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Get the lqr controller of the model (K)
        :param Q: state cost matrix, shape (state_dim, state_dim)
        :param R: control cost matrix, shape (control_dim, control_dim)
        :return: lqr controller, shape (batch_size, control_dim, state_dim)
        """
        K_list = []
        # for A_d, B_d in zip(self.A_d, self.B_d):  # TODO: 使用离散模型会导致计算结果失效，为什么？
        for A_d, B_d in zip(self.A, self.B):  # Apollo 使用连续模型进行计算
            K, S, E = lqr(A_d, B_d, Q, R)
            # print(A_d, B_d)
            # print("K = ", K)
            K_list.append(-K.copy())

        return np.stack(K_list)

    def get_controller_value(self, now_state: np.ndarray) -> np.ndarray:
        """
        Get the controller value of the model (u)
        :return: controller value, shape (batch_size, control_dim)
        """
        self.update_controller(now_state)
        # print(self.K)

        error_state = self.error_state[:, :, None]
        return np.squeeze(self.K @ error_state, axis=2)

    def get_action(self,
                   now_state: np.ndarray,
                   done: np.ndarray = None,
                   adaptive_delta: np.ndarray = None,
                   target_v: float = 3,
                   zero_Fxf: bool = False,
                   with_adaptive: bool = False,
                   action_normalize: bool = False) -> tuple:
        """
        Get the actions in lengthways and lateral
        :param now_state: current state, shape (batch_size(num not done), state_dim)
        :param done: whether each vehicle is done, shape (total_num_vehicles, )
        :param adaptive_delta: adaptive controller delta tracking result, shape (batch_size(num not done), lateral_control_dim(1))
        :param target_v: target speed
        :param zero_Fxf: whether to set Fxf to zero
        :param with_adaptive: whether to use adaptive controller
        :param action_normalize: whether to normalize the action

        :return: action, shape ( batch_size, total_control_dim( lateral(dim=1) and lengthways(dim=1) ) ) and delta_est, shape (batch_size, lateral_control_dim(1))
        """
        # update vehicle state parameters
        self.set_state_space(now_state)
        self.set_adaptive_param(adaptive_delta)

        vx = self.state_space[:, 3]
        lengthways_control_value = base_lengthways_control(vx, target_v) if not zero_Fxf else np.zeros(vx.shape)
        lateral_control_value = np.squeeze(self.get_controller_value(now_state), axis=1)

        if with_adaptive:
            lateral_control_value += np.squeeze(self.get_adjust_action(action_normalize, done), axis=1)

        if action_normalize:
            lateral_control_value /= self.action_high[0]
            lateral_control_value = np.clip(lateral_control_value, -1, 1)
            lengthways_control_value /= self.action_high[1]
            lengthways_control_value = np.clip(lengthways_control_value, -1, 1)

        return np.array([lateral_control_value, lengthways_control_value]).T.copy(), self.delta_est.copy()

    def error_state_update(self) -> np.ndarray:
        """
        Update the error state of the model (u) (mainly used for testing and visualization)
        :return: new error state, shape (batch_size, control_dim)
        """
        self.error_state = self.A_d @ self.error_state[:, :, None] + self.B_d @ self.K @ self.error_state[:, :, None]
        self.error_state = np.squeeze(self.error_state, axis=2)
        return self.error_state

    def get_adjust_action(self, action_normalize: bool = False, done_idx: np.ndarray = None) -> np.ndarray:
        """
        Get the RL controller tracking result
        :param action_normalize: whether the action is normalized
        :param done_idx: index of done vehicles, shape (total_num_vehicles, )

        :return: RL controller tracking result, shape (batch_size(num not done), lateral_control_dim(1))
        """
        if done_idx is None:
            done_idx = np.zeros(self.num_vehicles, dtype=bool)

        beta = np.squeeze(self.gamma * self.B_d.transpose(0, 2, 1) @ self.error_state[:, :, None], axis=2)

        delta_est_dot = - self.gamma * self.B_d.transpose(0, 2, 1) @ (self.A_d + self.B_d @ self.K) @ self.error_state[:, :, None]
        self.delta_est[~done_idx] = self.delta_est[~done_idx] + np.squeeze(delta_est_dot, axis=2) * self.dt

        adaptive_control_value = self.delta_est[~done_idx] + beta

        if action_normalize:
            adaptive_control_value /= self.action_high[0]
            adaptive_control_value = np.clip(adaptive_control_value, -1, 1)

        return - adaptive_control_value


if __name__ == '__main__':
    np.random.seed(0)

    Q = np.diag([1, 0, 1, 0])
    R = np.diag([1])

    random_seed = 5
    np.random.seed(random_seed)

    trace_path = "trace/sweep.npy"
    num_vehicles = 1

    env = SimEnv(trace_path, num_vehicles=num_vehicles)
    # controller = ClassicalController(trace_path, env.num_vehicles, env.state_space_aug.shape[1])
    controller = LinearTrackingAdaptiveController(trace_path, env.num_vehicles, Q, R)

    env.seed(random_seed)
    env.reset()
    state = env.state_space_aug.copy()
    state[:, 3] = 20
    show_animation = True

    # controller.get_action(state, target_v=3)
    # print(controller.error_state)
    #
    # for i in range(100):
    #     print(controller.error_state_update())
    # exit()

    max_step = 10000

    if show_animation:
        plt.ion()
        env.render()
        plt.pause(0.001)
        # plt.show()
    i = 0
    while True:
        print(i)
        print(state)
        action = controller.get_action(state[:, :state_space_aug_dim], target_v=3, with_adaptive=True)
        # print(action)
        # print(controller.get_controller_value(state[:, :state_space_aug_dim]))
        action = np.clip(action, env.action_space.low, env.action_space.high)
        # print(action)
        state = env.step(action, return_raw_state=True)
        # print(state[:, :4])

        if show_animation:
            env.render(show_dest_points_rate=False, quiver_scale=5, vehicle_view=True)

        i += 1
        if i >= max_step:
            break

        if env.done.all():
            break

    if show_animation:
        plt.ioff()