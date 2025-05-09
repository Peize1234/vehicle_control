import numpy as np
from MotionModel import BicycleModel
from Trace import Trace
import torch
import os
import matplotlib.pyplot as plt
import time
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ModelTraceInteractor(Trace, BicycleModel):
    def __init__(self, init_vehicle_state: np.ndarray, trace_path=None,
                 target_points=None, start_point_rate=None, end_point_rate=None, dt=0.01, seed=1, plot=False):
        super().__init__(target_points, start_point_rate, end_point_rate, dt, seed)
        super(Trace, self).__init__(init_vehicle_state)

        if trace_path is not None:
            self.load_trace(trace_path)

            self.curve_type_num = self.curve_type_num
            assert self.curve_type_num is not None, \
                "Please call Trace.generate_trace() or load_trace() before constructing ModelTraceInteractor"

            # shape = (curve_num, t_num, 2)
            self.curve = self.curve.copy()

        np.random.seed(0)
        self.color = plt.get_cmap('tab20')(np.linspace(0, 1, self.state_space.shape[0]))
        np.random.shuffle(self.color)

        self.fig, self.ax = None, None
        if plot:
            self.fig, self.ax = plt.subplots()

    def select_curve(self, start_points, normal_rates):
        batch_size = start_points.shape[0]
        curve_num = self.curve.shape[0]

        alphas = np.array([-normal_rates[:, 1], normal_rates[:, 0]]).T
        support_points_stack, split_indices = self.support_points_stack
        dis = alphas[:, None, :] @ (support_points_stack[None, :, :] - start_points[:, None, :]).transpose(0, 2, 1)
        dis = dis.squeeze()
        # TODO: 系统 bug？
        # dis_split = np.array(np.split(dis, split_indices, axis=1), dtype=object)

        dis_split = np.array(np.split(dis.T, split_indices, axis=0), dtype=object)
        points_split = np.array(np.split(support_points_stack, split_indices, axis=0), dtype=object)
        curve_points = [None for i in range(self.curve_type_num)]
        curve_points_split_num = [None for i in range(self.curve_type_num)]
        for point_type in range(self.curve_type_num):
            if np.all(~(self.point_type_idx == point_type)):
                continue
            dis_batch = np.stack(dis_split[self.point_type_idx == point_type], axis=0)
            points_batch = np.repeat(np.stack(points_split[self.point_type_idx == point_type], axis=0)[None, ...],
                                     batch_size, axis=0)

            dis_batch_label = ~(np.all(dis_batch >= 0, axis=1) | np.all(dis_batch <= 0, axis=1))

            if np.any(dis_batch_label):
                selected_points = points_batch[dis_batch_label.T]
                points_split_num = np.sum(dis_batch_label, axis=0)

                curve_points[point_type] = selected_points
                curve_points_split_num[point_type] = points_split_num

        return curve_points, curve_points_split_num

    @staticmethod
    def get_rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).transpose(2, 0, 1)

    def transform_curve(self, batch_points, gc, normal_vec):
        batch_points -= gc[:, None, :]
        theta = np.arctan2(normal_vec[:, 1], normal_vec[:, 0])
        rotation_matrix = self.get_rotation_matrix(theta)
        batch_points = batch_points @ rotation_matrix
        return batch_points, rotation_matrix

    @staticmethod
    def batch_root_p1(batch_y):
        assert np.any((batch_y[:, 0] - batch_y[:, 1] < 1e-5) == False)

        t = batch_y[:, 0] / (batch_y[:, 0] - batch_y[:, 1])
        valid_idx = (t >= 0) & (t <= 1)
        return t[None, :], valid_idx

    @staticmethod
    def batch_root_p2(batch_y):
        a = batch_y[:, 0] + batch_y[:, 2] - 2 * batch_y[:, 1]
        b = 2 * (batch_y[:, 1] - batch_y[:, 0])
        c = batch_y[:, 0]

        delta = b ** 2 - 4 * a * c
        t1 = (- b + np.sqrt(delta)) / (2 * a)
        t2 = (- b - np.sqrt(delta)) / (2 * a)

        valid_idx = (delta >= 0) & (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
        return np.vstack((t1, t2)).T, valid_idx

    @staticmethod
    def batch_root_p3(batch_y):
        # print(3)
        pass

    def _build_t_vector_p3(self, t_seq):
        build_t_vector_diff0 = lambda t: np.array([1, t, t ** 2, t ** 3])
        build_t_vector_diff1 = lambda t: np.array([0, 1, 2 * t, 3 * t ** 2])
        build_t_vector_diff2 = lambda t: np.array([0, 0, 2, 6 * t])
        build_t_vector = [build_t_vector_diff0, build_t_vector_diff1, build_t_vector_diff2]

        t_vector = np.stack([np.vstack([builder(t) for t in t_seq]) for builder in build_t_vector]).transpose(1, 0, 2)
        return t_vector

    def _build_t_vector_p2(self, t_seq):
        build_t_vector_diff0 = lambda t: np.array([1, t, t ** 2])
        build_t_vector_diff1 = lambda t: np.array([0, 1, 2 * t])
        build_t_vector_diff2 = lambda t: np.array([0, 0, 2])
        build_t_vector = [build_t_vector_diff0, build_t_vector_diff1, build_t_vector_diff2]

        t_vector = np.stack([np.vstack([builder(t) for t in t_seq]) for builder in build_t_vector]).transpose(1, 0, 2)
        return t_vector

    def _build_t_vector(self, t_seq, curve_type):
        if curve_type == 0:
            return None
        elif curve_type == 1:
            return self._build_t_vector_p2(t_seq)
        elif curve_type == 2:
            return self._build_t_vector_p3(t_seq)
        else:
            raise ValueError("Invalid curve type, must be 0, 1 or 2")


    def get_intersect_points(self, batch_transformed_points, rotation_matrix, point_type):
        if point_type == 0:
            root = self.batch_root_p1(batch_transformed_points[..., 1])
        elif point_type == 1:
            root, valid_idx = self.batch_root_p2(batch_transformed_points[..., 1])
        elif point_type == 2:
            root = self.batch_root_p3(batch_transformed_points[..., 1])
        else:
            raise ValueError("Invalid point type, must be 0, 1 or 2")

        print(root.shape)

    def _get_site_points(self, site="center"):
        if site == "center":
            site_points = self.vehicle_center
        elif site == "front":
            site_points = self.front_points
        elif site == "rear":
            site_points = self.rear_points
        else:
            raise ValueError("Invalid site, must be 'center', 'front' or 'rear'")
        return site_points

    def _vehicle_closed_trace_point(self, site="center"):
        """
        寻找与车辆当前位置最近的点，并返回该点所在的曲线的坐标（以曲线参数形式表示）

        :param site: 用于计算最近点位置的点，"center"表示车辆中心，"front"表示车辆前轮中心，"rear"表示车辆后轮中心

        :return: 最近点所在的曲线的坐标，以及该点在曲线上的位置，分辨率为 1/t_num
        """
        site_points = self._get_site_points(site)

        vehicle_expand = site_points[:, None, None, :]
        curve_expand = self.curve[None, :, :, :]

        distance = np.linalg.norm(vehicle_expand - curve_expand, axis=-1).reshape(site_points.shape[0], -1)
        min_idx = np.argmin(distance, axis=-1)

        t_num = self.curve.shape[1]
        curve_idx = min_idx // t_num
        t_idx = np.mod(min_idx, t_num)

        return curve_idx, t_idx

    def _batch_opt_step_p1(self, selected_vehicle_center, selected_support_point):
        assert selected_vehicle_center.shape[0] == selected_support_point.shape[0]

        # (x1 - x2)^T @ (x - x0) == 0 | x = alpha * x1 + (1 - alpha) * x2
        # => (x1 - x2)^T @ (alpha * x1 + (1 - alpha) * x2 - x0) == 0
        # => alpha = (x1 - x2)^T @ (x - x0) / (x1 - x2)^T @ (x1 - x2) | alpha = 1 - t
        sub_12 = selected_support_point[:, 0:1, :] - selected_support_point[:, 1:2, :]
        sub_02 = selected_vehicle_center - selected_support_point[:, 1:2, :]

        factor1 = sub_12 @ sub_12.transpose(0, 2, 1)
        factor2 = sub_12 @ sub_02.transpose(0, 2, 1)
        t_accurate = 1 - (factor2 / factor1).squeeze()
        within_range = (t_accurate > 0) & (t_accurate < 1)
        end = within_range

        return t_accurate, within_range, end

    def _batch_opt_step_base(self, selected_vehicle_center, selected_support_point,
                           M_B, t_vectors, t_old, accuracy):
        assert selected_vehicle_center.shape[0] == selected_support_point.shape[0] == \
               t_vectors.shape[0], "Batch size must be equal"

        batch_G = selected_support_point.transpose(0, 2, 1)  # (N, 3, 2)
        batch_M_B = M_B[None, ...]
        batch_t_vectors = t_vectors.transpose(0, 2, 1)
        batch_vehicle_center = selected_vehicle_center.transpose(0, 2, 1)

        GMB = batch_G @ batch_M_B

        # 需要求解的公式： ft = (G @ M_B @ t_vector_diff)^T @ (G @ M_B @ t_vector - x0) == 0
        # 上式根据曲线外一点与曲线上最近点的关系得到(曲线外一点与曲线上最近点的连线与此点的法向量垂直)
        # 贝塞尔曲线的表示方法：x(t) = G(几何矩阵) @ M_B(贝塞尔基矩阵) @ t_vector(t的各次幂构成的列向量)
        ft = ((GMB @ batch_t_vectors[:, :, 1:2]).transpose(0, 2, 1) @
              (GMB @ batch_t_vectors[:, :, 0:1] - batch_vehicle_center)).squeeze()
        ft_dot = ((GMB @ batch_t_vectors[:, :, 2:3]).transpose(0, 2, 1) @
                  (GMB @ batch_t_vectors[:, :, 0:1] - batch_vehicle_center) +
                  (GMB @ batch_t_vectors[:, :, 1:2]).transpose(0, 2, 1) @
                  (GMB @ batch_t_vectors[:, :, 1:2])).squeeze()

        # 牛顿法迭代公式 x(t+1) = x(t) - f(x(t)) / f'(x(t))
        factor = ft / ft_dot
        t_update = t_old - factor
        # if t_update[0] > 1:
        #     print(t_update)
        within_range = (t_update >= 0) & (t_update <= 1)
        end = np.abs(factor) < accuracy

        return t_update, within_range, end

    def _batch_opt_step_p2(self, selected_vehicle_center, selected_support_point,
                           t_vectors, t_old, accuracy):
        M_B = self.B2.copy()
        return self._batch_opt_step_base(selected_vehicle_center, selected_support_point,
                                         M_B, t_vectors, t_old, accuracy)

    def _batch_opt_step_p3(self, selected_vehicle_center, selected_support_point,
                           t_vectors, t_old, accuracy):
        M_B = self.B3.copy()
        return self._batch_opt_step_base(selected_vehicle_center, selected_support_point,
                                         M_B, t_vectors, t_old, accuracy)

    def _batch_opt_step(self, selected_vehicle_center, selected_support_point,
                        t_vectors, t_old, curve_type_idx, accuracy):
        if curve_type_idx == 0:
            return self._batch_opt_step_p1(selected_vehicle_center, selected_support_point)
        elif curve_type_idx == 1:
            return self._batch_opt_step_p2(selected_vehicle_center, selected_support_point,
                                           t_vectors, t_old, accuracy)
        elif curve_type_idx == 2:
            return self._batch_opt_step_p3(selected_vehicle_center, selected_support_point,
                                           t_vectors, t_old, accuracy)
        else:
            raise ValueError("Invalid curve type index, must be 0, 1 or 2")

    def _update_item(self, t_update, curve_idx, end_flags, select_idx, curve_types,
                    support_points_per_vehicle, support_points, curve_types_per_vehicle):
        sub_idx = t_update < 0
        add_idx = t_update > 1

        # 如果车辆所对应的最近点的t_update小于0，且车辆所在的曲线索引为0，表示此车辆对应在轨迹上的最近点超过了曲线的起点，需要强制结束
        sub_force_end_idx = sub_idx & (curve_idx == 0)

        # 如果车辆所对应的最近点的t_update大于1，且车辆所在的曲线索引为曲线的最后一个，表示此车辆对应在轨迹上的最近点超过了曲线的终点，需要强制结束
        add_force_end_idx = add_idx & (curve_idx == support_points.shape[0] - 1)

        # curve_idx 需要 减少 的车辆的索引（排除强制结束的车辆）
        select_sub_idx = select_idx & sub_idx & ~sub_force_end_idx

        # curve_idx 需要 增加 的车辆的索引（排除强制结束的车辆）
        select_add_idx = select_idx & add_idx & ~add_force_end_idx

        # 更新 curve_types_per_vehicle 、 support_points_per_vehicle 和 curve_idx
        if np.any(select_sub_idx):
            curve_types_per_vehicle[select_sub_idx] = curve_types[
                curve_idx[select_sub_idx] - 1]
            support_points_per_vehicle[select_sub_idx] = support_points[
                curve_idx[select_sub_idx] - 1]

            curve_idx[select_sub_idx] -= 1
            t_update[select_sub_idx] = 1
        if np.any(select_add_idx):
            curve_types_per_vehicle[select_add_idx] = curve_types[
                curve_idx[select_add_idx] + 1]
            support_points_per_vehicle[select_add_idx] = support_points[
                curve_idx[select_add_idx] + 1]

            curve_idx[select_add_idx] += 1
            t_update[select_add_idx] = 0

        # 对于强制结束(继续更新会超出曲线范围)的车辆，更新 curve_idx 、 t_update 和 end_flags
        t_update[select_idx & sub_force_end_idx] = 0
        curve_idx[select_idx & sub_force_end_idx] = 0

        t_update[select_idx & add_force_end_idx] = 1
        curve_idx[select_idx & add_force_end_idx] = support_points.shape[0] - 1

        end_flags[select_idx & (sub_force_end_idx | add_force_end_idx)] = True

    def _optimize_point_position(self, init_curve_idx, init_t_idx, accuracy, max_iter=10, site="center"):
        """
        优化车辆当前位置最近的点的位置

        :param init_curve_idx: 各个车辆当前位置最近的点（低分辨率）所在的曲线的索引，形状为 (batch_size(vehicle_num),)
        :param init_t_idx: 各个车辆当前位置最近的点在曲线上的位置(参数表示)，形状为 (batch_size(vehicle_num),)
        :param accuracy: 迭代精度
        :param max_iter: 最大迭代次数
        :param site: 用于计算最近点位置的点，"center"表示车辆中心，"front"表示车辆前轮中心，"rear"表示车辆后轮中心

        :return: 优化后的曲线索引，优化后的曲线参数
        """
        site_points = self._get_site_points(site)

        t_per_vehicle = self.t[init_t_idx]
        curve_idx = init_curve_idx

        curve_types = self.point_type_idx
        support_points = self.support_points

        # 选择每个车辆最近点的曲线的类型
        curve_types_per_vehicle = curve_types[init_curve_idx]
        # 选择每个车辆最近点的曲线的控制点(支持点)
        support_points_per_vehicle = support_points[init_curve_idx]

        end_flags = np.zeros(init_curve_idx.shape[0], dtype=bool)
        select_idx_change = [True, True, True]
        for type_idx in range(self.curve_type_num):
            for _ in range(max_iter):
                if select_idx_change[type_idx]:
                    # 根据曲线类型对每个不同类型的曲线做不同的处理
                    select_idx = curve_types_per_vehicle == type_idx
                    if np.all(~select_idx):
                        continue

                    # 对每个车辆最近点所对应的曲线，不同类型的控制点处理方式不同
                    # 如果 curve_types 全为一种类型，需要做debug
                    selected_support_points = np.stack(support_points_per_vehicle[select_idx], axis=0).astype(np.float32)

                    # 对每辆车进行同样的索引操作
                    selected_vehicle_center = site_points[select_idx][:, None, :]

                    select_idx_change[type_idx] = False

                # debug
                # print(selected_vehicle_center_tensor.shape)
                # print(selected_support_point_tensor.shape)
                # self.show_connect_lines(self.get_vehicle_closed_trace_point(), show=False)
                # for vehicle_center_ in selected_vehicle_center_tensor:
                #     self.ax.scatter(vehicle_center_.cpu().numpy()[0, 0], vehicle_center_.cpu().numpy()[0, 1], s=20)
                #     self.ax.scatter(selected_support_point_tensor[:, 0, 0].cpu().numpy(),
                #                     selected_support_point_tensor[:, 0, 1].cpu().numpy(), s=50)
                # plt.show()

                # 如果此次迭代没有完成所有点的优化，则继续迭代
                if not np.all(end_flags[select_idx]):
                    # 索引：select_idx & ~end_flags 表示的是 curve type 类型对应的曲线上未优化完成的点
                    # （~end_flags 的作用：从节约计算资源的角度考虑，已经优化完成的点不需要再参与优化）
                    # 对 selected_vehicle_center_tensor 和 selected_support_point_tensor 的索引是为了保持后续计算维度的一致
                    t_per_vehicle[select_idx & ~end_flags], within_range, end = (
                        self._batch_opt_step(selected_vehicle_center[~end_flags[select_idx]],
                                             selected_support_points[~end_flags[select_idx]],
                                             self._build_t_vector(t_per_vehicle[select_idx & ~end_flags],
                                                                  type_idx), t_per_vehicle[select_idx & ~end_flags],
                                             type_idx, accuracy))

                    end_flags[select_idx & ~end_flags] = end & within_range

                    # 如果 within_range 不全为 True，说明 t_update 超出了曲线的范围，需要调整
                    if not np.all(within_range):
                        self._update_item(t_per_vehicle, curve_idx, end_flags, select_idx, curve_types,
                                          support_points_per_vehicle, support_points, curve_types_per_vehicle)
                        select_idx_change[type_idx] = True
                    else:
                        if np.all(end):
                            break

        if not np.all(end_flags):
            warnings.warn("Optimization not converged, some points may not be optimized")

        return curve_idx, t_per_vehicle

    def get_vehicle_closed_trace_point(self, site="center", accuracy=None):
        """
        寻找与车辆当前位置最近的点，并返回该点所在的曲线的坐标

        :param site: 用于计算最近点位置的点，"center"表示车辆中心，"front"表示车辆前轮中心，"rear"表示车辆后轮中心
        :param accuracy: 获取曲线上的点的精度为多少

        :return: 如果 accuracy 为 None，返回该点在曲线上的位置（实际坐标），分辨率为 1 / t_num
                 如果 accuracy 为 数值，返回该点在曲线上的位置（实际坐标），分辨率为 accuracy
                 shape: (num_not_done, ...)
        """
        assert accuracy is None or isinstance(accuracy, (int, float)), "Invalid accuracy type, must be None or number"
        assert site in ["center", "front", "rear"], "Invalid site type, must be 'center', 'front' or'rear'"

        curve_idx, t_idx = self._vehicle_closed_trace_point(site=site)
        if accuracy is None:
            closed_points = self.curve[curve_idx, t_idx]
            return closed_points, curve_idx, self.t[t_idx]
        else:
            opt_curve_num, opt_t = self._optimize_point_position(curve_idx, t_idx, accuracy, site=site)
            closed_points = self.get_trace_points_from_curveidx_and_t(opt_curve_num, opt_t)
            return closed_points, opt_curve_num, opt_t

    def _iter_split(self, curve_idx, start_t, gap_distance, requires_n_points):
        """
        用于分解原问题为多个子问题，每个子问题的描述为：在给定曲线(索引)上，给定起始点和需要积分的距离，得到终点的坐标和方向向量
        此函数的作用为根据所提供的参数，计算各个子问题的曲线索引、起始点和需要积分的长度
        子问题划分的方式：按照 requires_n_points 进行划分，每个子问题只寻找其中一个点的位置(但是以batch的方式)，并计算其方向向量
        所以子问题的数量(即返回列表的长度)为 requires_n_points

        :param curve_idx: 起始点所在曲线的索引，形状为 (batch_size,)
        :param start_t: 起始点在曲线上的位置(参数表示)，形状为 (batch_size,)
        :param gap_distance: 积分间隔距离
        :param requires_n_points: 所需获取的点数

        :return: 按照 requires_n_points 进行划分的子问题的曲线索引、起始点和需要积分的长度, 形状均为 (batch_size, requires_n_points)
        """
        # 计算此线段的剩余长度
        surplus_length = self.get_batch_line_length(curve_idx, start_t, np.ones_like(curve_idx, dtype=np.float64))

        # 维护需要分解的子问题的曲线索引、起始点和需要积分的长度 shape: (batch_size, requires_n_points)
        surplus_length = np.repeat(surplus_length[:, None], requires_n_points, axis=1)
        split_curve_idx = np.repeat(curve_idx[:, None], requires_n_points, axis=1)
        split_start_t = np.repeat(start_t[:, None], requires_n_points, axis=1)
        split_gap_distance = np.repeat(np.array([[(i + 1) * gap_distance for i in range(requires_n_points)]]),
                                       curve_idx.shape[0], axis=0)

        # 循环查找需要分解的子问题的曲线索引、起始点和需要积分的长度
        # （主要考虑积分长度超出剩余曲线长度的情况，此时需要对曲线索引、起始点和需要积分的长度进行调整）
        for n in range(1, requires_n_points + 1):
            # 获取 积分长度超出剩余曲线长度 的曲线索引
            over_bound_idx = surplus_length[:, n - 1] < gap_distance * n
            # 使用循环判断，是为了防止修改后仍然有部分曲线的剩余长度小于 gap_distance * n 的情况
            while np.any(over_bound_idx):
                # 调整曲线索引、起始点和需要积分的长度
                split_curve_idx[over_bound_idx, n - 1] += 1
                split_start_t[over_bound_idx, n - 1] = 0
                split_gap_distance[over_bound_idx, n - 1] = gap_distance * n - surplus_length[over_bound_idx, n - 1]
                surplus_length[over_bound_idx, n - 1] += self.get_batch_line_length(split_curve_idx[over_bound_idx, n - 1],
                                                                                    np.zeros_like(
                                                                                        start_t[over_bound_idx],
                                                                                        dtype=np.float64),
                                                                                    np.ones_like(
                                                                                        start_t[over_bound_idx],
                                                                                        dtype=np.float64))
                over_bound_idx[over_bound_idx] = surplus_length[over_bound_idx, n - 1] < gap_distance * n

        return split_curve_idx, split_start_t, split_gap_distance

    def get_n_points_rate_along_trace(self, curve_idx, start_t, gap_distance=0.5, requires_n_points=5):
        """
        根据所给出起始点对轨迹进行积分，得到每隔 gap_distance 距离的点的坐标和方向向量，总共获取的点数为 requires_n_points

        :param curve_idx: 起始点所在曲线的索引，形状为 (batch_size,)
        :param start_t: 起始点在曲线上的位置(参数表示)，形状为 (batch_size,)
        :param gap_distance: 积分间隔距离
        :param requires_n_points: 所需获取的点数

        :return: 获取的点的坐标和方向向量，形状为 (batch_size, requires_n_points, 2) 和 (batch_size, requires_n_points, 2)
        """

        split_curve_idx, split_start_t, split_gap_distance = self._iter_split(curve_idx,
                                                                              start_t,
                                                                              gap_distance,
                                                                              requires_n_points)

        # (batch_size, requires_n_points, 2)
        end_points, end_t = self.get_batch_points_by_length(split_curve_idx, split_start_t, split_gap_distance)
        rate_norm = self.get_rate_by_curve_idx_and_t(split_curve_idx, end_t)

        return end_points, rate_norm

    def transfer_points_rates_to_vehicle_coordinate(self, points, rates, site="center"):
        """
        将点坐标和方向向量转换到车辆坐标系下

        :param points: 点坐标，形状为 (vehicle_num, points_num, 2) or None
        :param rates: 点方向向量，形状为 (vehicle_num, points_num, 2) or None
        :param site: 用于计算车辆中心的点，"center"表示车辆中心，"front"表示车辆前轮中心，"rear"表示车辆后轮中心

        :return: 点坐标和方向向量，转换到车辆坐标系下后的坐标和方向向量，形状为 (vehicle_num, points_num, 2) or None
        """
        assert site in ["center", "front", "rear"], "Invalid site type, must be 'center', 'front' or 'rear'"
        # (num_vehicle, 2)
        site_coord = self._get_site_points(site)

        # (num_vehicle, 2, 2)
        rotation_matrix = self._build_rotation_matrix(self.state_space[~self.done, 2])

        points_trans = None
        rate_trans = None
        if points is not None:
            points_trans = rotation_matrix[:, None, :, :] @ (points[..., None] - site_coord[:, None, :, None])
            points_trans = points_trans.squeeze(-1)
        if rates is not None:
            rate_trans = rotation_matrix[:, None, :, :] @ rates[..., None]
            rate_trans = rate_trans.squeeze(-1)

        return points_trans, rate_trans

    def _build_rotation_matrix(self, angles):
        """
        构建旋转矩阵

        :param angles: 旋转角度 shape: (batch_size, )

        :return: 旋转矩阵 shape: (batch_size, 2, 2)
        """
        return np.array([[np.cos(angles), np.sin(angles)],
                         [-np.sin(angles), np.cos(angles)]]).transpose(2, 0, 1)


    def show_vehicle(self, delta, show=False):
        self.show_vehicle_state(delta=delta, ax=self.ax, show=show)

    def show_points(self, points, show=False):
        self.ax.scatter(points[~self.done, 0], points[~self.done, 1], color=self.color[~self.done], s=10, marker='o')
        if show:
            plt.show()

    def show_connect_lines(self, points_per_vehicle, show=False):
        """
        绘制各个车辆的最近轨迹点之间的连线

        :param points_per_vehicle: 各个车辆的最近轨迹点，形状为 (batch_size, 2)
        :param show: 是否显示
        """
        vehicle_center = self.state_space[~self.done, :2].copy()
        self.ax.plot(np.hstack((vehicle_center[:, 0:1], points_per_vehicle[:, 0:1])).T,
                     np.hstack((vehicle_center[:, 1:2], points_per_vehicle[:, 1:2])).T,
                     color=self.color[0])
        if show:
            plt.show()

    def show_n_points(self, points, show=False):
        """
        绘制 每个车辆的 n多个 目的地点

        :param points: 各个车辆的 n多个 目的地点，形状为 (batch_size, n, 2)
        :param show: 是否显示
        """
        for idx in range(points.shape[1]):
            self.show_points(points[:, idx, :], show=False)

        if show:
            plt.show()

    def show_n_rate(self, start_points, rates, quiver_scale=1, show=False):
        assert start_points.shape == rates.shape, "start_points and rates must have the same shape"

        for idx in range(rates.shape[1]):
            start_point = start_points[~self.done, idx, :]
            rate = rates[~self.done, idx, :]
            self.ax.quiver(start_point[:, 0], start_point[:, 1], rate[:, 0], rate[:, 1],
                           angles='xy', scale_units='xy', scale=quiver_scale)

        if show:
            plt.show()


if __name__ == '__main__':
    import time
    np.random.seed(0)

    interactor = ModelTraceInteractor(np.random.randn(50, 6) * 40, "./trace/trace1.npy")
    # interactor.done[:25] = True
    # print(interactor.trace.B1)

    # t1 = time.time()
    close_points, curve_idx, t = interactor.get_vehicle_closed_trace_point(accuracy=1e-6)
    points = np.array([[0, 0], [11, 0]])
    rates = np.array([[1.0, 0.0], [0.0, 1.0]])
    interactor.transfer_points_rates_to_vehicle_coordinate(points, rates, site="center")
    # print(close_points.shape, curve_idx.shape, t.shape)
    # exit()
    # t2 = time.time()
    # print("time cost:", t2 - t1)
    # impact_points, impact_rate = interactor.get_n_points_rate_along_trace(curve_idx, t)
    #
    # # interactor.highlight_points(close_points)
    # interactor.show_trace(show=False)
    # interactor.show_connect_lines(close_points, show=False)
    # interactor.show_n_points(impact_points, show=False)
    # interactor.show_n_rate(impact_points, impact_rate, show=True)

