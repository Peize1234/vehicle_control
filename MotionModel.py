import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

state_space_dim = 6
state_space_aug_dim = 8
u_ref = 0.7

def sgn(x):
    return np.where(x >= 0, 1, -1)


class BicycleModel:
    def __init__(self, init_state_space, m=1500, Iz=2250, a=1.04, b=1.42, h=0.6, g=9.81, Cxf=2.5e3,
                 Cxr=2.5e3, Cyf=160e3, Cyr=180e3, sigma_f=0.3, sigma_r=0.3):
        """
        :param init_state_space: 初始状态空间，shape=(sequence_num, state_space_dim)
        :param m: 车轮质量 取值一般在1000～5000kg
        :param Iz: 车轮转向惯量 取值一般在1000～5000kg*m^2
        :param a: 前轮距 取值一般在0.5～2m
        :param b: 后轮距 取值一般在0.5～2m
        :param h: 车辆中心高度 取值一般在0.5～1m
        :param g: 重力加速度 取值一般在9.81～10.2
        :param Cxf, Cxr 轮胎纵向（x方向）刚度 取值一般在1000～5000N/m^2
        :param Cyf, Cyr 轮胎横向（y方向）刚度 取值一般在10000～50000N/m^2
        :param sigma_f, sigma_r 轮胎松弛长度 取值一般在0.1～1
        """

        self.m = m
        self.Iz = Iz
        self.a = a
        self.b = b
        self.L = a + b  # 车辆长轴距
        self.h = h
        self.g = g
        self.Cxf = Cxf
        self.Cxr = Cxr
        self.Cyf = Cyf
        self.Cyr = Cyr
        self.sigma_f = sigma_f
        self.sigma_r = sigma_r

        if init_state_space is not None:
            # x, y, phi, Ux, Uy, r
            if init_state_space.shape[-1] == state_space_dim:
                self.state_space = init_state_space
                self.state_space_aug = np.concatenate((self.state_space,
                                                       np.zeros((self.state_space.shape[0], 2))), axis=-1)  # alpha_f, alpha_r
            elif init_state_space.shape[-1] == state_space_aug_dim:
                self.state_space = init_state_space[:, :state_space_dim].copy()
                self.state_space_aug = init_state_space  # alpha_f, alpha_r
            else:
                raise ValueError(f"state_space shape should be "
                                 f"(sequence_num, {state_space_dim}) or (sequence_num, {state_space_aug_dim})")

            self.done = np.zeros(self.state_space.shape[0], dtype=bool)
        else:
            self.state_space = None
            self.state_space_aug = None
            self.done = None

        self.ax = None

    def set_state_space(self, state_space_aug, done=None):
        assert state_space_aug.shape[-1] == state_space_aug_dim, "state_space shape should be (sequence_num, 8)"
        self.state_space = state_space_aug[:, :state_space_dim].copy()
        self.state_space_aug = state_space_aug  # alpha_f, alpha_r
        if done is None:
            self.done = np.zeros(self.state_space.shape[0], dtype=bool)
        else:
            self.done = done

    def get_Fy(self, alpha, C, u, Fz):
        Fy = np.where(np.abs(alpha) < np.arctan(3 * u * Fz / C),
                      - C * np.tan(alpha) + C ** 2 / (3 * u * Fz) * np.abs(np.tan(alpha)) * np.tan(alpha) - \
                      C ** 3 / (27 * u ** 2 * Fz ** 2) * np.tan(alpha) ** 3,
                      - u * Fz * sgn(alpha))
        return Fy

    @property
    def alpha_f_plus_delta(self):
        Ux = self.state_space[:, 3]
        Uy = self.state_space[:, 4]
        r = self.state_space[:, 5]
        return (Uy + self.a * r) / Ux

    @property
    def alpha_r(self):
        Ux = self.state_space[:, 3]
        Uy = self.state_space[:, 4]
        r = self.state_space[:, 5]
        return (Uy - self.b * r) / Ux

    @property
    def V(self):
        return np.sqrt(self.state_space[:, 3] ** 2 + self.state_space[:, 4] ** 2)

    def predict(self, delta, Fxf, u):
        not_done = ~self.done
        phi = self.state_space[not_done, 2]
        Ux = self.state_space[not_done, 3]
        Uy = self.state_space[not_done, 4]
        r = self.state_space[not_done, 5]
        alpha_f = self.state_space_aug[not_done, 6]
        alpha_r = self.state_space_aug[not_done, 7]
        # delta = delta[not_done]
        # Fxf = Fxf[not_done]

        L = self.a + self.b

        ax = r * Uy + Fxf * np.cos(delta) / self.m

        Fzf = self.b / L * self.m * self.g - self.h / L * self.m * ax
        Fzr = self.a / L * self.m * self.g + self.h / L * self.m * ax

        Fyf = self.get_Fy(alpha_f, self.Cyf, u, Fzf)
        Fyr = self.get_Fy(alpha_r, self.Cyr, u, Fzr)

        X_dot = Ux * np.cos(phi) - Uy * np.sin(phi)
        Y_dot = Ux * np.sin(phi) + Uy * np.cos(phi)
        phi_dot = r.copy()

        Ux_dot = Fxf * np.cos(delta) / self.m + r * Uy
        Uy_dot = (Fyr + Fyf * np.cos(delta) + Fxf * np.sin(delta)) / self.m - r * Ux
        r_dot = (self.a * Fyf * np.cos(delta) + self.a * Fxf * np.sin(delta) - self.b * Fyr) / self.Iz

        alpha_f_dot = self.V[not_done] / self.sigma_f * (np.arctan2(Uy + self.a * r, Ux) - delta - alpha_f)
        alpha_r_dot = self.V[not_done] / self.sigma_r * (np.arctan2(Uy - self.b * r, Ux) - alpha_r)

        return X_dot, Y_dot, phi_dot, Ux_dot, Uy_dot, r_dot, alpha_f_dot, alpha_r_dot

    def step_RungeKutta(self, delta, Fxf, u, dt=0.01):
        not_done = ~self.done
        raw_state_space_aug = self.state_space_aug.copy()
        z1 = np.vstack(self.predict(delta, Fxf, u)).T

        self.state_space_aug[not_done] = raw_state_space_aug[not_done] + dt / 2 * z1
        z2 = np.vstack(self.predict(delta, Fxf, u)).T

        self.state_space_aug[not_done] = raw_state_space_aug[not_done] + dt / 2 * z2
        z3 = np.vstack(self.predict(delta, Fxf, u)).T

        self.state_space_aug[not_done] = raw_state_space_aug[not_done] + dt * z3
        z4 = np.vstack(self.predict(delta, Fxf, u)).T

        self.state_space_aug[not_done] = (raw_state_space_aug[not_done] + dt * (z1 + 2 * z2 + 2 * z3 + z4) / 6).copy()
        self.state_space = self.state_space_aug[:, :self.state_space.shape[1]].copy()

        return self.state_space

    def step_eular(self, delta, Fxf, u, dt=0.01):
        state_space_aug_dot = np.vstack(self.predict(delta, Fxf, u)).T
        self.state_space_aug = (self.state_space_aug + dt * state_space_aug_dot).copy()
        self.state_space = self.state_space_aug[:, :self.state_space.shape[1]].copy()

        return self.state_space

    def get_stepper(self, method='RungeKutta'):
        if method == 'RungeKutta':
            return self.step_RungeKutta
        elif method == 'eular':
            return self.step_eular
        else:
            raise ValueError("method should be 'RungeKutta' or 'eular'")

    def step_once(self, delta, Fxf, u=0.8, dt=0.01, method='RungeKutta', return_aug_space=False):
        """
        :param delta: 车辆转向角(逆时针旋转为正)，shape=(num_vehicle,)，一般范围在-40°～40°
        :param Fxf: 前轮摩擦力，shape=(num_vehicle,)， 一般范围在-4000～4000N
        :param u: 摩擦系数，shape=(num_vehicle,) or scalar, 取值一般在0.3～1
        :param dt: 时间步长
        :param method: 预测方法，'RungeKutta' or 'eular'
        :param done: 是否停止更新车辆的状态
        :param return_aug_space: 是否返回增广状态空间

        :return: 状态空间，shape=(num_vehicle(not done), state_space_dim / state_space_aug_dim)
        """
        assert self.state_space is not None

        if self.state_space.shape[0] != 1:
            assert delta.shape[0] == Fxf.shape[0] == self.state_space[~self.done].shape[0]

        self.get_stepper(method)(delta, Fxf, u, dt)
        # 确保 phi 的取值范围在 0 到 2*pi 之间
        self.state_space[:, 2] = np.fmod(self.state_space[:, 2], 2 * np.pi)
        self.state_space_aug[:, 2] = np.fmod(self.state_space_aug[:, 2], 2 * np.pi)

        assert np.any(np.isfinite(self.state_space_aug) == True) and np.any(np.isnan(self.state_space_aug) == False)

        if return_aug_space:
            return self.state_space_aug[~self.done]
        return self.state_space[~self.done]

    def step_n(self, delta, Fxf, u, dt=0.01, method='RungeKutta', return_aug_space=False, add_raw_state_space=False):
        """
        :param delta: 车辆转向角(逆时针旋转为正)，shape=(num_vehicle, sequence_num)
        :param Fxf: 前轮受力，shape=(num_vehicle, sequence_num)
        :param u: 摩擦系数，shape=(num_vehicle, sequence_num) or scalar, 取值一般在0.3～1
        :param dt: 时间步长
        :param method: 预测方法，'RungeKutta' or 'eular'
        :param return_aug_space: 是否返回增广状态空间
        :param add_raw_state_space: 是否返回原始状态空间(step 0 时的初始状态)

        :return: 状态空间，shape=(num_vehicle, sequence_num, state_space_dim)
        """
        assert ((len(delta.shape) == len(Fxf.shape) == len(u.shape) == 2) and
                (delta.shape[1] == Fxf.shape[1] == u.shape[1]) and
                (delta.shape[0] == Fxf.shape[0] == u.shape[0] == self.state_space.shape[0]) and
                (self.state_space is not None))

        out_list = [self.state_space_aug.copy() if return_aug_space else self.state_space.copy()]
        for i in range(delta.shape[1]):
            delta_i = delta[:, i]
            Fxf_i = Fxf[:, i]
            u_i = u[:, i]
            out_list.append(self.step_once(delta_i, Fxf_i, u_i, dt, method, return_aug_space).copy())

        if add_raw_state_space:
            return np.stack(out_list, axis=1)
        return np.stack(out_list[1:], axis=1)

    @property
    def normal_vector(self):
        return np.array([-np.sin(self.state_space[:, 2]), np.cos(self.state_space[:, 2])]).T

    @property
    def vehicle_center(self):
        return self.state_space[~self.done, :2].copy()

    @property
    def forward_norm_vector(self):
        angle = self.state_space[~self.done, 2]
        return np.array([np.cos(angle), np.sin(angle)]).T

    @property
    def rear_points(self):
        return self.state_space[~self.done, :2] - np.array([self.b * np.cos(self.state_space[~self.done, 2]),
                                                            self.b * np.sin(self.state_space[~self.done, 2])]).T

    @property
    def front_points(self):
        return self.state_space[~self.done, :2] + np.array([self.a * np.cos(self.state_space[~self.done, 2]),
                                                            self.a * np.sin(self.state_space[~self.done, 2])]).T

    def show_vehicle_state(self, delta=0, ax=None, show=False):
        if ax is None:
            _, self.ax = plt.subplots()
            ax = self.ax

        xy = self.state_space[~self.done, :2]
        phi = self.state_space[~self.done, 2]
        delta = delta[~self.done]
        xy_rear = xy - np.array([self.b * np.cos(phi), self.b * np.sin(phi)]).T
        xy_front = xy + np.array([self.a * np.cos(phi), self.a * np.sin(phi)]).T
        tire_r_half = (self.a + self.b) / 4 / 2

        front_tire_xy_front = xy_front + np.array([tire_r_half * np.cos(phi + delta),
                                                   tire_r_half * np.sin(phi + delta)]).T
        front_tire_xy_rear = xy_front - np.array([tire_r_half * np.cos(phi + delta),
                                                  tire_r_half * np.sin(phi + delta)]).T

        ax.plot(np.hstack((xy_rear[:, 0:1], xy_front[:, 0:1])).T,
                np.hstack((xy_rear[:, 1:2], xy_front[:, 1:2])).T, color='r')
        ax.plot(np.hstack((front_tire_xy_rear[:, 0:1], front_tire_xy_front[:, 0:1])).T,
                np.hstack((front_tire_xy_rear[:, 1:2], front_tire_xy_front[:, 1:2])).T, color='b')
        ax.scatter(xy[:, 0], xy[:, 1], color='g', s=1)
        ax.scatter(xy_rear[:, 0], xy_rear[:, 1], color='g', s=1)
        ax.scatter(xy_front[:, 0], xy_front[:, 1], color='r', s=1)

        if show:
            ax.xlim([-10, 20])
            ax.ylim([-10, 20])
            ax.gca().set_aspect('equal', adjustable='box')
            plt.show()


def draw_model_state():
    def draw_model_state_once(model):
        xy = model.state_space[:, :2]
        phi = model.state_space[:, 2:3]
        xy_rear = xy - np.hstack([model.b * np.cos(phi), model.b * np.sin(phi)])
        xy_front = xy + np.hstack([model.a * np.cos(phi), model.a * np.sin(phi)])
        tire_r_half = (model.a + model.b) / 4 / 2

        front_tire_xy_front = xy_front + np.hstack(
            [tire_r_half * np.cos(phi + delta_arr[i][:, None]), tire_r_half * np.sin(phi + delta_arr[i][:, None])])
        front_tire_xy_rear = xy_front - np.hstack(
            [tire_r_half * np.cos(phi + delta_arr[i][:, None]), tire_r_half * np.sin(phi + delta_arr[i][:, None])])

        plt.xlim([0, 30])
        plt.ylim([-10, 20])
        plt.gca().set_aspect('equal', adjustable='box')

        for j in range(xy.shape[0]):
            plt.plot([xy_front[j, 0], xy_rear[j, 0]], [xy_front[j, 1], xy_rear[j, 1]], color='r')
            plt.plot([front_tire_xy_rear[j, 0], front_tire_xy_front[j, 0]],
                     [front_tire_xy_rear[j, 1], front_tire_xy_front[j, 1]],
                     color='b')
            plt.plot(xy[j, 0], xy[j, 1], 'o', color='g')
            plt.plot(xy_rear[j, 0], xy_rear[j, 1], 'o', color='g')
            plt.plot(xy_front[j, 0], xy_front[j, 1], 'o', color='r')
            # plt.plot(front_tire_xy_rear[0], front_tire_xy_rear[1], 'o', color='g')
            # plt.plot(front_tire_xy_front[0], front_tire_xy_front[1], 'o', color='g')
        print(model.state_space[0])
        plt.pause(0.01)

    init_state_space = np.array([[1, 1, 0 * np.pi / 180, 1, 0, 0]])
    init_state_space = np.repeat(init_state_space, 50, axis=0)
    N = 1000

    # delta_arr = np.ones(N) * 30 * np.pi / 180
    delta_arr = np.random.uniform(-np.pi / 2, np.pi / 2, size=(N, init_state_space.shape[0]))

    # Fxf_arr = np.linspace(0, 2000, N)
    Fxf_arr = np.random.uniform(0, 2000, size=(N, init_state_space.shape[0]))

    model1 = BicycleModel(init_state_space)
    model2 = BicycleModel(init_state_space)

    plt.ion()
    state_list1 = [init_state_space]
    state_list2 = [init_state_space]
    for i in range(N):
        print(i)
        model1.step_RungeKutta(delta_arr[i], Fxf_arr[i], 1)
        # model2.step_eular(delta_arr[i], Fxf_arr[i], 1)

        state_list1.append(model1.state_space[0].copy())
        # state_list2.append(model2.state_space[0].copy())

        draw_model_state_once(model1)
        # draw_model_state_once(model2)
        print("=" * 100)
        plt.clf()
    plt.ioff()


def generate_data():
    # state_space space limit
    phi_limit = [0, 2 * np.pi]
    Ux_limit = [-30, 30]
    Uy_limit = [-9.8, 9.8]
    r_limit = [-30 * np.pi / 180, 30 * np.pi / 180]

    # control space limit
    delta_limit = [-40 * np.pi / 180, 40 * np.pi / 180]
    Fxf_limit = [-2000, 2000]
    u_limit = [0, 1]

    sequence_num = 10000
    sequence_length = 10

    reference_u = u_ref

    np.random.seed(0)

    for epoch in tqdm(range(1000)):
        init_state_space = np.random.uniform(low=np.array([0, 0, phi_limit[0], Ux_limit[0], Uy_limit[0], r_limit[0]]),
                                             high=np.array([0, 0, phi_limit[1], Ux_limit[1], Uy_limit[1], r_limit[1]]),
                                             size=(sequence_num, state_space_dim))

        # inference once
        # delta_random_sampling = np.random.uniform(delta_limit[0], delta_limit[1], sequence_num)
        # Fxf_random_sampling = np.random.uniform(Fxf_limit[0], Fxf_limit[1], sequence_num)
        # u_random_sampling = np.random.uniform(u_limit[0], u_limit[1], sequence_num)

        # inference multiple times
        delta_random_sampling = np.random.uniform(delta_limit[0], delta_limit[1], (sequence_num, sequence_length))
        Fxf_random_sampling = np.random.uniform(Fxf_limit[0], Fxf_limit[1], (sequence_num, sequence_length))
        u_random_sampling = np.random.uniform(u_limit[0], u_limit[1], (sequence_num, 1)).repeat(sequence_length, axis=1)
        u_random_sampling_aug = u_random_sampling + np.random.uniform(-0.1, 0.1, (sequence_num, sequence_length))

        # inference once
        # sim_output = BicycleModel(init_state_space).step_once(delta_random_sampling, Fxf_random_sampling, u_random_sampling)

        # inference multiple times
        sim_output = BicycleModel(init_state_space).step_n(delta_random_sampling, Fxf_random_sampling,
                                                           u_random_sampling_aug)
        sim_reference = BicycleModel(init_state_space).step_n(delta_random_sampling, Fxf_random_sampling,
                                                              reference_u * np.ones_like(u_random_sampling_aug))

        assert np.any(np.isfinite(sim_output) == True) and np.any(np.isnan(sim_output) == False)

        if epoch == 0:
            print("input shape:", init_state_space.shape)
            print("reference shape: ", sim_reference.shape)
            print("output shape: ", sim_output.shape)

            print("Fxf shape: ", Fxf_random_sampling.shape)
            print("delta shape: ", delta_random_sampling.shape)
            print("u shape: ", u_random_sampling.shape)

        np.save(f"./sim_model_data/sim_init_state{epoch}.npy", init_state_space)
        np.save(f"./sim_model_data/sim_reference{epoch}.npy", sim_reference)
        np.save(f"./sim_model_data/sim_output{epoch}.npy", sim_output)

        np.save(f"./sim_model_data/sim_Fxf{epoch}.npy", Fxf_random_sampling)
        np.save(f"./sim_model_data/sim_delta{epoch}.npy", delta_random_sampling)
        np.save(f"./sim_model_data/sim_u{epoch}.npy", u_random_sampling)
        np.save(f"./sim_model_data/sim_u_aug{epoch}.npy", u_random_sampling_aug)


if __name__ == '__main__':

    generate_data()

    # draw_model_state()



