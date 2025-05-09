import numpy as np
from Model_Trace_Interactor import ModelTraceInteractor
from env import SimEnv
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy as sp


class ClassicalController(ModelTraceInteractor):
    def __init__(self, trace_path: str, num_vehicles=50, state_space_dim=6):
        """
        参考网址：https://blog.csdn.net/ChenGuiGan/article/details/116459381
        """
        super().__init__(np.zeros((num_vehicles, state_space_dim)), trace_path)

    def get_action(self, state, method="pure_pursuit", zero_Fxf=False, **kwargs):
        """
        Get the action based on the current state and method.

        ::param state: The current state of the environment. shape: (num_vehicles, state_space_dim)
        ::param method: The method used to get the action. "pure_pursuit" or "".

        ::return: The action for each vehicle. shape: (num_vehicles, action_space_dim)
        """
        controller = getattr(self, method, None)
        if method is None:
            raise ValueError("Invalid method")
        self.set_state_space(state)
        control_signal = controller(**kwargs)
        if zero_Fxf:
            control_signal[:, 1] = 0
        return control_signal

    def base_lengthways_control(self, target_speed=3):
        """
        Get the action based on the current state using base lengthways control method.

        :return: The Fxf action for each vehicle. shape: (num_vehicles, action_space_dim)
        """
        return np.where(self.state_space[:, 3] < target_speed, 4000, -4000)

    def pure_pursuit(self):
        """
        Get the action based on the current state using pure pursuit method.

        :return: The delta and Fxf action for each vehicle. shape: (num_vehicles, action_space_dim)
        """

        _, closed_curve_idx, closed_t = self.get_vehicle_closed_trace_point(accuracy=1e-6)
        target_points = self.get_n_points_rate_along_trace(closed_curve_idx, closed_t,
                                                           gap_distance=2, requires_n_points=1)[0].squeeze(1)
        rear_points = self.rear_points

        vehicle_forward_vector = self.forward_norm_vector
        r2g_vector = target_points - rear_points
        ld = np.linalg.norm(r2g_vector, axis=1)
        r2g_norm_vector = r2g_vector / ld.reshape(-1, 1)
        alpha = np.arccos(np.sum(vehicle_forward_vector * r2g_norm_vector, axis=1))
        vector_cross = np.cross(vehicle_forward_vector, r2g_norm_vector)
        alpha = np.where(vector_cross > 0, alpha, -alpha)

        return np.array([np.arctan(2 * self.L * np.sin(alpha) / ld),
                         self.base_lengthways_control()]).T

    def rear_wheel_feedback(self, k_phi=1, k_e=0.5):
        """
        Get the action based on the current state using rear wheel feedback method.
        note: 本控制方案需要修改车辆的初始位置，使得后轮尽量在在轨迹起点前方

        :return: The delta and Fxf action for each vehicle. shape: (num_vehicles, action_space_dim)
        """
        vehicle_rear_points = self.rear_points
        closed_points, closed_curve_idx, closed_t = self.get_vehicle_closed_trace_point(site="rear", accuracy=1e-6)
        vehicle_forward_vector = self.forward_norm_vector
        target_rates = self.get_n_points_rate_along_trace(closed_curve_idx, closed_t,
                                                          gap_distance=0, requires_n_points=1)[1].squeeze(1)
        curvature = self.get_curvature_by_curve_idx_and_t(closed_curve_idx, closed_t)

        e = np.linalg.norm(vehicle_rear_points - closed_points, axis=1)
        phi_e = np.arccos(np.sum(vehicle_forward_vector * target_rates, axis=1))
        phi_e = np.where(np.cross(target_rates, vehicle_forward_vector) > 0, phi_e, -phi_e) + 1e-5  # 避免出现 nan
        ks = curvature
        vr = self.state_space[~self.done, 3].copy()

        factor1 = vr * ks * np.cos(phi_e) / (1 - ks * e)  # 无方向基础量
        factor2 = - k_phi * np.abs(vr) * phi_e  # 向偏角误差更小的方向(phi_e 可以控制 factor2 的方向)
        factor3 = - k_e * vr * np.sin(phi_e) * e / phi_e  # 向距离误差更小的方向(本身恒为负(与建模有关)，需要做后续处理)
        # 确保 factor3 的方向总是朝着距离误差更小的方向(参考：https://blog.csdn.net/weixin_42301220/article/details/125003918)
        factor3 = np.where(np.cross(target_rates, vehicle_rear_points - closed_points) > 0, factor3, -factor3)

        w = factor1 + factor2 + factor3

        return np.array([np.arctan2(w * self.L, vr),
                         self.base_lengthways_control()]).T

    def front_wheel_feedback(self, k=1):
        vehicle_front_points = self.front_points
        closed_points, closed_curve_idx, closed_t = self.get_vehicle_closed_trace_point(site="front", accuracy=1e-6)
        vehicle_forward_vector = self.forward_norm_vector
        target_rates = self.get_n_points_rate_along_trace(closed_curve_idx, closed_t,
                                                          gap_distance=0, requires_n_points=1)[1].squeeze(1)

        e = np.linalg.norm(vehicle_front_points - closed_points, axis=1)
        e = np.where(np.cross(target_rates, vehicle_front_points - closed_points) > 0, e, -e)

        phi_e = np.arccos(np.sum(vehicle_forward_vector * target_rates, axis=1))
        phi_e = np.where(np.cross(target_rates, vehicle_forward_vector) > 0, phi_e, -phi_e) + 1e-5  # 避免出现 nan

        return np.array([np.arctan2(- k * e, self.state_space[~self.done, 3]) - phi_e,
                         self.base_lengthways_control()]).T

    def lqr(self):
        pass

    def build_error_matrix(self):
        A = np.zeros((np.sum(~self.done), 4, 4))
        A[:, 0, 1] = 1
        A[:, 2, 3] = 1
        A[:, 1, 1] = (- 2 * self.Cyf + 2 * self.Cyr) / (self.m * (self.state_space[~self.done, 3] + 1e-5))
        A[:, 1, 2] = (2 * self.Cyf + 2 * self.Cyr) / self.m
        A[:, 1, 3] = (- 2 * self.a * self.Cyf + 2 * self.b * self.Cyr) / (self.m * (self.state_space[~self.done, 3] + 1e-5))
        A[:, 3, 1] = (- 2 * self.a * self.Cyf - 2 * self.b * self.Cyr) / (self.Iz * (self.state_space[~self.done, 3] + 1e-5))
        A[:, 3, 2] = (2 * self.a * self.Cyf - 2 * self.b * self.Cyr) / self.Iz
        A[:, 3, 3] = (- 2 * (self.a ** 2) * self.Cyf + 2 * (self.b ** 2) * self.Cyr) / (self.Iz * (self.state_space[~self.done, 3] + 1e-5))

        B = np.zeros((np.sum(~self.done), 4, 1))
        B[:, 1, 0] = 2 * self.Cyf / self.m
        B[:, 3, 0] = 2 * self.a * self.Cyf / self.Iz

        return A, B


    def discretize(self, A, B, dt=0.1):
        A_trans_list = []
        B_trans_list = []
        for a, b in zip(A, B):
            A_trans = expm(a * dt)
            A_trans_list.append(A_trans)

            eigvals, eigvecs = np.linalg.eig(A_trans)
            t = sp.symbols('t')
            eigvals_exp_A_dt_vec_sp = sp.Array([sp.exp(eigvals[i] * t) for i in range(len(eigvals))])
            eigvals_exp_A_dt_diag_sp = sp.diag(*eigvals_exp_A_dt_vec_sp)

            exp_A_dt_sp = sp.Matrix(eigvecs) * eigvals_exp_A_dt_diag_sp * sp.Matrix(eigvecs).inv()
            factor = exp_A_dt_sp * b

            B_trans_list.append([sp.integrate(factor[i, 0], (t, 0, dt)) for i in range(factor.shape[0])])

        return np.array(A_trans_list).reshape(A.shape), np.array(B_trans_list).reshape(B.shape)


if __name__ == '__main__':
    random_seed = 5
    np.random.seed(random_seed)

    trace_path = "trace/sweep.npy"
    num_vehicles = 1

    env = SimEnv(trace_path, num_vehicles=num_vehicles)
    controller = ClassicalController(trace_path, env.num_vehicles, env.state_space_aug.shape[1])

    env.seed(random_seed)
    state = env.reset()
    show_animation = True

    max_step = 10000

    if show_animation:
        plt.ion()
        env.render()
        plt.pause(0.001)
        # plt.show()
    i = 0
    while True:
        print(i)
        action = controller.get_action(state[:, :env.state_space_aug.shape[1]], method="front_wheel_feedback")
        action = np.clip(action, env.action_space.low, env.action_space.high)
        # print(action)
        state = env.step(action, return_raw_state=True)
        print(state[:, :4])

        if show_animation:
            env.render(show_dest_points_rate=False, quiver_scale=5, vehicle_view=True)

        i += 1
        if i >= max_step:
            break

        if env.done.all():
            break

    if show_animation:
        plt.ioff()