import numpy as np
import torch
from scipy.special import comb
import matplotlib.pyplot as plt
from gauss_legendre_integral_points import xk, Ak
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def bezier_basis_matrix(n):
    """
    构造贝塞尔基矩阵，其中矩阵的每一行是一个贝塞尔基函数的多项式系数。

    参数:
    n (int): Bézier曲线的阶数 (n+1 个控制点)

    返回:
    matrix (ndarray): 贝塞尔基矩阵，维度为 (n+1, n+1)
    """
    matrix = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        # 构造贝塞尔基函数的系数（多项式系数）
        # (1 - t)^(n-i) * t^i 的多项式展开系数
        for j in range(n + 1):
            # 每个基函数的系数是二项式系数 * (1-t) 和 t 的组合
            matrix[i, j] = comb(n, i) * comb(n - i, j) * (-1) ** (n - i - j)

    return np.flip(matrix, axis=1)


class Trace:
    def __init__(self, target_points=None, start_point_rate=None, end_point_rate=None, dt=0.01, seed=1):
        """
        :param target_points: 控制点，shape 为 (N, 2) N >= 3
        :param start_point_rate: 起点方向向量，shape 为 (2,)
        :param end_point_rate: 终点方向向量，shape 为 (2,)
        """
        if target_points is not None:
            assert target_points.shape[0] >= 3, "控制点数至少为 3"

        self.B1 = bezier_basis_matrix(1)
        self.B2 = bezier_basis_matrix(2)
        self.B3 = bezier_basis_matrix(3)

        self.target_points = target_points
        self.start_point_rate = start_point_rate
        self.end_point_rate = end_point_rate

        self.dt = dt
        self.t = np.arange(0, 1 + dt, self.dt)

        self.points_club = []
        self.curve = None
        self.support_points = None

        self.p1_points = None
        self.p2_points = None
        self.p3_points = None

        # 曲线类型的索引
        self.point_type_idx = None
        self.curve_type_num = None

        self.mode = ["p2_p3", "p3"]
        np.random.seed(seed)

        self.ax = None
        self.color_init = False
        if self.target_points is not None:
            self.color_trace = plt.get_cmap('tab20')(np.linspace(0, 1, target_points.shape[0]))
            np.random.shuffle(self.color_trace)
            self.color_init = True

    def _ut1(self, t):
        return np.vstack((np.ones(len(t)), t))

    def _ut2(self, t):
        return np.vstack((np.ones(len(t)), t, t ** 2))

    def _ut2_diff(self, t):
        return np.stack((np.zeros(t.shape), np.ones(t.shape), 2 * t), axis=1)

    def _ut2_diff_diff(self, t):
        return np.stack((np.zeros(t.shape), np.zeros(t.shape), 2 * np.ones(t.shape)), axis=1)

    def _ut3(self, t):
        return np.vstack((np.ones(len(t)), t, t ** 2, t ** 3))

    def _ut3_diff(self, t):
        return np.stack((np.zeros(t.shape), np.ones(t.shape), 2 * t, 3 * t ** 2), axis=1)

    def _ut3_diff_diff(self, t):
        return np.stack((np.zeros(t.shape), np.zeros(t.shape), 2 * np.ones(t.shape), 6 * t), axis=1)

    @property
    def support_points_stack(self):
        assert self.support_points is not None, "请先调用 generate_support_points 方法生成支持点 or load support_points"
        return np.vstack(self.support_points), np.cumsum([p.shape[0] for p in self.support_points])[:-1]

    def generate_trace(self, mode="p2_p3"):
        """
        生成通过所有控制点的贝塞尔轨迹

        参数:
        :param mode: "p2_p3" 或 "p3"，p2_p3 表示插值点为两个或三个（自动判断），p3 表示插值点为三个
        """
        assert mode in self.mode, "mode 参数错误，请选择 p2_p3 或 p3"
        if mode == "p2_p3":
            self.curve_type_num = 3
        else:
            self.curve_type_num = 1

        # 归一化初始方向向量
        start_point_rate_norm = self.start_point_rate / np.linalg.norm(self.start_point_rate)
        end_point_rate_norm = self.end_point_rate / np.linalg.norm(self.end_point_rate)

        # 计算两点连线的方向向量
        target_points_diff = self.target_points[1:] - self.target_points[:-1]
        target_points_diff_norm = target_points_diff / np.linalg.norm(target_points_diff, axis=1, keepdims=True)
        target_points_diff_norm_vertical = np.hstack([-target_points_diff_norm[:, 1:], target_points_diff_norm[:, 0:1]])

        # 计算两个点的中点
        center_point = (self.target_points[1:] + self.target_points[:-1]) / 2

        # 计算内点的方向向量
        inner_point_rate = (target_points_diff_norm[1:] + target_points_diff_norm[:-1]) / 2

        # 所有点的方向向量，target_point是所有的输入的控制点
        target_point_rate = np.vstack((start_point_rate_norm, inner_point_rate, end_point_rate_norm))

        # 控制点及其方向向量  [p1; p2; ...; pn] -> [p1; p2; p2; p3; p3; ...; pn-1; pn-1; pn]
        target_point_extend = self.get_array_inner_extend(self.target_points)
        target_points_rate_extend = self.get_array_inner_extend(target_point_rate)

        # 两点中点及其方向向量  [p1; p2; ...; pn] -> [p1; p1; p2; p2; p3; p3; ...; pn-1; pn-1; pn; pn]
        center_point_extend = np.repeat(center_point, 2, axis=0)
        center_point_rate_vertical_extend = np.repeat(target_points_diff_norm_vertical, 2, axis=0)

        # 构造 y
        y_extend = target_point_extend - center_point_extend

        # 构造 A 和 b （批量求解）
        target_points_rate_extend = target_points_rate_extend[:, :, np.newaxis]
        center_point_rate_vertical_extend = center_point_rate_vertical_extend[:, :, np.newaxis]
        y_extend = y_extend[:, :, np.newaxis]

        # 构造 A 和 b
        # x1(x2) + t1(t2) * r1(r2) = xc + m * v (x1, r1, x2, r2, xc, v is vector, t1, t2, m is scalar)
        A = np.concatenate((-target_points_rate_extend, center_point_rate_vertical_extend), axis=2)
        b = y_extend.copy()

        # 求解 A @ x = b (shape 为 ((num_target_points-1) * 2, 2, 1) = (求解方程组个数, (t(控制点方程参数), m(中点垂线方程参数)), 1))
        res = np.linalg.inv(A) @ b

        # 过两点中点，方向为两点连线的垂线与两点方向向量的交点参数取值
        # reshape后，shape 为 (num_target_points-1, 2, 2) = (控制点个数-1(中点个数), 2(中点垂线与两条控制点方程的交点), (t(控制点方程参数), m(中点垂线方程参数)))
        t = res.reshape(-1, 2, 2)[:, :, 0]
        m = res.reshape(-1, 2, 2)[:, :, 1]

        # 大于0表示两交点处于两点连线的同一侧，小于零表示两交点处于两点连线的反方向
        # 等于零表示其中一个交点或两个交点在两点的连线上，这种情况在后续判断中处理
        insert_point_num = np.where(np.prod(m, axis=1) > 0, 1, 2) if mode == "p2_p3" else np.ones(m.shape[0], dtype=int) * 2

        # 每两个点之间插入一个或两个点
        # 记录每两个点之间插入的点的类型
        self.point_type_idx = np.zeros(len(insert_point_num), dtype=int)
        # 循环构造 points_club
        for i, num in enumerate(insert_point_num):
            # 如果需要插入一个点
            if num == 1:
                # x1 + t1 * r1 = x2 - t2 * r2 (x1, r1, x2, r2 is vector, t1, t2 is scalar)
                A1 = np.vstack((target_point_rate[i], target_point_rate[i + 1])).T
                b1 = self.target_points[i + 1] - self.target_points[i]

                # 矩阵奇异，表示插入点和控制点的连线共线，此时无需插入中间点
                if np.abs(np.linalg.det(A1)) < 1e-6:
                    self.points_club.append(np.vstack((self.target_points[i], self.target_points[i + 1])))
                    continue

                # 求解方程组，得到插入点（插入点为两控制点对应参数方程（所表示直线）的交点）
                insert_point = (np.linalg.inv(A1) @ b1)[0] * target_point_rate[i] + self.target_points[i]
                self.points_club.append(np.vstack((self.target_points[i], insert_point, self.target_points[i + 1])))
                self.point_type_idx[i] = 1
            # 如果需要插入两个点
            else:
                # 计算需要插入的点的位置
                insert_points = center_point[i] + m[i][:, None] * np.repeat(target_points_diff_norm_vertical[i:i + 1],
                                                                            2, axis=0)

                # 所插入的两点在两控制点的连线的连线上
                if np.all(np.abs(m[i]) < 1e-6):
                    self.points_club.append(np.vstack((self.target_points[i], self.target_points[i + 1])))
                    continue

                # 如果第一条线段的方程参数小于0，表示交点在给定向量方向的负向，需要调整（第一条线段的方程参数必须为正）
                if t[i, 0] < 0:
                    insert_points[0] = self.target_points[i] - t[i, 0] * target_point_rate[i]
                self.points_club.append(np.vstack((self.target_points[i], insert_points, self.target_points[i + 1])))
                self.point_type_idx[i] = 2

        # 每条单个的贝塞尔曲线
        points_club = np.array(self.points_club, dtype=object)

        # 插入点分别为 0(p1)， 1(p2)， 2(p3) 的曲线
        self.p1_points = np.stack(points_club[self.point_type_idx == 0], axis=0) if np.any(self.point_type_idx == 0) else None
        self.p2_points = np.stack(points_club[self.point_type_idx == 1], axis=0) if np.any(self.point_type_idx == 1) else None
        self.p3_points = np.stack(points_club[self.point_type_idx == 2], axis=0) if np.any(self.point_type_idx == 2) else None

        # 根据控制点得到曲线
        p1_curve = self.get_p1_curve(self.p1_points)
        p2_curve = self.get_p2_curve(self.p2_points)
        p3_curve = self.get_p3_curve(self.p3_points)

        # 合并曲线
        self.curve = np.zeros((self.target_points.shape[0] - 1, self.t.shape[0], 2))
        self.curve[self.point_type_idx == 0] = p1_curve
        self.curve[self.point_type_idx == 1] = p2_curve
        self.curve[self.point_type_idx == 2] = p3_curve

        # 记录支持点
        self.support_points = points_club.copy()

        self.add_end_curve()
        return p1_curve, p2_curve, p3_curve

    def add_end_curve(self, length=100):
        """
        在所生成的路径的终点额外添加一条直线，用于防止后续计算过程中的索引超界问题
        """
        curve_end_point = self.target_points[-1].copy()
        curve_end_point_rate = self.end_point_rate.copy()
        line_end_point = curve_end_point + length * curve_end_point_rate
        add_p1_point = np.vstack([curve_end_point, line_end_point])
        line = self.get_p1_curve(add_p1_point[None, :, :])

        self.target_points = np.append(self.target_points, line_end_point[None, :], axis=0)
        self.points_club.append(add_p1_point)
        self.support_points = np.array(self.points_club, dtype=object)
        self.curve = np.append(self.curve, line, axis=0)
        self.p1_points = np.append(self.p1_points, add_p1_point[None, :, :],
                                   axis=0) if self.p1_points is not None else add_p1_point[None, :, :].copy()
        self.point_type_idx = np.append(self.point_type_idx, 0)

    def get_p1_curve(self, p1_points):
        if p1_points is None:
            return None
        return (p1_points.transpose(0, 2, 1) @ self.B1[None, :, :] @ self._ut1(self.t)[None, :, :]).transpose(0, 2, 1)

    def get_p2_curve(self, p2_points):
        if p2_points is None:
            return None
        return (p2_points.transpose(0, 2, 1) @ self.B2[None, :, :] @ self._ut2(self.t)[None, :, :]).transpose(0, 2, 1)

    def get_p3_curve(self, p3_points):
        if p3_points is None:
            return None
        return (p3_points.transpose(0, 2, 1) @ self.B3[None, :, :] @ self._ut3(self.t)[None, :, :]).transpose(0, 2, 1)

    @staticmethod
    def get_array_inner_extend(points):
        if points.shape[0] > 2:
            new_arr = np.concatenate([points[:1], np.repeat(points[1:-1], 2, axis=0), points[-1:]], axis=0)
        else:
            new_arr = points.copy()
        return new_arr

    def _batch_trace_points_from_support_points_and_t(self, support_points, t, curve_type=0):
        """
        通过支持点和 t 得到曲线上对应的点

        :param support_points: 控制点，shape 为 (batch_size, p+1, 2)
        :param t: 控制点方程参数，shape 为 (batch_size,)
        :param curve_type: 曲线类型，0, 1 或 2

        :return: 批量的曲线上对应的点，shape 为 (batch_size, 2)
        """
        assert support_points.shape[0] == t.shape[0], "curve_idx 和 t 必须有相同的长度"

        batch_G = support_points.transpose(0, 2, 1)
        if curve_type == 0:
            batch_M_B = self.B1[None, :, :]
            batch_t = self._ut1(t).T[:, :, None]
        elif curve_type == 1:
            batch_M_B = self.B2[None, :, :]
            batch_t = self._ut2(t).T[:, :, None]
        elif curve_type == 2:
            batch_M_B = self.B3[None, :, :]
            batch_t = self._ut3(t).T[:, :, None]
        else:
            raise ValueError("p must be 1, 2 or 3")

        batch_points = (batch_G @ batch_M_B @ batch_t).squeeze()
        return batch_points

    def get_trace_points_from_curveidx_and_t(self, batch_curve_idx, batch_t):
        """
        根据 curve_idx 和 t 得到曲线上对应的点

        :param batch_curve_idx: 批量的 curve_idx，shape 为 (batch_size,)
        :param batch_t: 批量的 t，shape 为 (batch_size,)

        :return: 批量的曲线上对应的点，shape 为 (batch_size, 2)
        """
        assert batch_curve_idx.shape[0] == batch_t.shape[0], "curve_idx 和 t 必须有相同的长度"

        points = np.zeros((batch_curve_idx.shape[0], 2))
        batch_curve_types = self.point_type_idx[batch_curve_idx]
        batch_support_points = self.support_points[batch_curve_idx]
        for curve_type in range(self.curve_type_num):
            select_idx = batch_curve_types == curve_type
            # 如果没有 curve_type 类型的曲线，则跳过
            if not np.any(select_idx):
                continue

            selected_support_points = np.stack(batch_support_points[select_idx])
            selected_t = batch_t[select_idx]

            points[select_idx] = self._batch_trace_points_from_support_points_and_t(selected_support_points, selected_t,
                                                                                    curve_type=curve_type)
        return points

    def _build_integrand_variable(self, t1, t2):
        """
        构建积分变量，t1, t2 -> in_var, factor1, factor2
        """
        # (batch_size, )
        factor1 = (t2 - t1) / 2
        factor2 = (t1 + t2) / 2

        # (batch_size, xk.shape[0]) 每个 curve_idx 对应的自变量不同
        in_var = factor1[:, None] * xk[None, :] + factor2[:, None]
        return in_var, factor1, factor2

    def _curve_integrand_func(self, support_points, in_var, curve_type=1):
        """
        曲线积分的被积函数：f(t) = sqrt(dx^2 + dy^2)

        :param support_points: 控制点，shape 为 (batch_size, p+1, 2)
        :param in_var: 积分变量，shape 为 (batch_size, xk.shape[0](不可省略))
        :param curve_type: 曲线类型，1 或 2

        :return: 曲线积分的被积函数值，shape 为 (batch_size, xk.shape[0])
        """
        support_points = support_points.transpose(0, 2, 1)

        # self._ut2_diff(in_var) -> (batch_size, p+1, xk.shape[0])
        # batch_dxy -> (batch_size, 2, xk.shape[0])
        if curve_type == 1:
            batch_dxy = support_points @ self.B2[None, :, :] @ self._ut2_diff(in_var)
        elif curve_type == 2:
            batch_dxy = support_points @ self.B3[None, :, :] @ self._ut3_diff(in_var)
        else:
            raise ValueError("curve_type must be 1 or 2")

        return np.sqrt(batch_dxy[:, 0, :] ** 2 + batch_dxy[:, 1, :] ** 2)

    def _line_integral_p2_p3(self, curve_idx, t1, t2, curve_type=1):
        """
        对二阶和三阶的曲线，使用7点Gauss-Legendre积分法计算曲线的长度(截断误差f^(14))
        """
        assert curve_type in [1, 2], "curve_type 必须为 1 或 2"

        # (batch_size, p+1, 2)
        support_point = np.stack(self.support_points[curve_idx]).astype(np.float32)

        # in_var -> (batch_size, xk.shape[0)
        # factor1 -> (batch_size,)
        # factor2 -> (batch_size,)
        in_var, factor1, factor2 = self._build_integrand_variable(t1, t2)
        # (batch_size, xk.shape[0])
        batch_fk = self._curve_integrand_func(support_point, in_var, curve_type=curve_type)

        result = np.sum(batch_fk * Ak[None, :], axis=1) * factor1
        return result

    def _line_integral_p1(self, curve_idx, t1, t2):
        """
        通过计算直线的长度进行积分
        """
        support_points = np.stack(self.support_points[curve_idx]).astype(np.float32)
        proportion = t2 - t1

        total_length = np.sqrt(np.sum((support_points[:, 0, :] - support_points[:, 1, :]) ** 2, axis=1))

        return total_length * proportion

    def _line_length_by_curve_type(self, curve_idx, t1, t2, curve_type=0):
        if curve_type == 0:
            length = self._line_integral_p1(curve_idx, t1, t2)
        elif curve_type == 1:
            length = self._line_integral_p2_p3(curve_idx, t1, t2, curve_type=1)
        elif curve_type == 2:
            length = self._line_integral_p2_p3(curve_idx, t1, t2, curve_type=2)
        else:
            raise ValueError("curve_type 必须为 0, 1 或 2")

        return length

    def get_batch_line_length(self, curve_idx, t1, t2):
        """
        根据 curve_idx, t1, t2 得到对应曲线上一段的长度

        :param curve_idx: 批量的 curve_idx，shape 为 (batch_size,)
        :param t1: 批量的 t1，shape 为 (batch_size,)
        :param t2: 批量的 t2，shape 为 (batch_size,)

        :return: 批量的曲线上一段的长度，shape 为 (batch_size,)
        """
        assert np.all(t1 <= t2), "t1 必须小于等于 t2"
        assert curve_idx.shape[0] == t1.shape[0] == t2.shape[0], "curve_idx, t1, t2 必须有相同的长度"

        batch_length = np.zeros_like(curve_idx, dtype=np.float64)

        for curve_type in range(self.curve_type_num):
            select_idx = self.point_type_idx[curve_idx] == curve_type
            if np.any(select_idx):
                batch_length[select_idx] = self._line_length_by_curve_type(curve_idx[select_idx],
                                                                           t1[select_idx],
                                                                           t2[select_idx],
                                                                           curve_type=curve_type)
        return batch_length

    def _get_points_by_length_p1(self, curve_idx, start_t, length):
        """
        获取直线段上与给定点相距为 length 的点（给定点前面的点，所以只有一个）

        :param curve_idx: 曲线索引， shape 为 (batch_size,)
        :param start_t: 起点参数， shape 为 (batch_size,)
        :param length: 距离， shape 为 (batch_size,)

        :return: 直线段上与给定点相距为 length 的点，shape 为 (batch_size, 2)
        """
        support_points = np.stack(self.support_points[curve_idx]).astype(np.float32)

        # ||x0 - x1|| = d , x0 = (1 - t0) * p1 + t0 * p2, x1 = (1 - t) * p1 + t * p2
        # -> ||x0 - x1|| = ||(t - t0) * (p1 - p2)|| == d
        # -> t = d / ||p1 - p2|| + t0  (t > t0, 绝对值省去)
        end_t = start_t + length / np.linalg.norm(support_points[:, 0, :] - support_points[:, 1, :], axis=1)
        return self._batch_trace_points_from_support_points_and_t(support_points, end_t, curve_type=0), end_t

    def _get_points_by_length_p2_p3(self, curve_idx, start_t, length, curve_type=1, max_iter=10, accuracy=1e-6):
        """
        获取二阶或三阶贝塞尔曲线段上与给定点相距为 length 的点（给定点前面的点，所以只有一个）
        二阶或三阶的贝塞尔曲线的长度计算公式：
            L = int_x0^x1 sqrt(dx^2 + dy^2) dt
        直接积分计算困难，采用牛顿法进行迭代计算(TODO:对于二阶贝塞尔曲线，上式存在解析解，可以尝试进行解析计算)

        :param curve_idx: 曲线索引， shape 为 (batch_size,)
        :param start_t: 起点参数， shape 为 (batch_size,)
        :param length: 距离， shape 为 (batch_size,)
        :param curve_type: 曲线类型，1 或 2
        :param max_iter: 最大迭代次数
        :param accuracy: 迭代计算的精度

        :return: 二阶或三阶贝塞尔曲线段上与给定点相距为 length 的点，shape 为 (batch_size, 2)
        """
        assert curve_type in [1, 2], "curve_type 必须为 1 或 2"

        support_points = np.stack(self.support_points[curve_idx]).astype(np.float32)
        end_t = start_t.copy()

        end_flags = np.zeros_like(start_t, dtype=bool)
        ft = np.empty_like(end_flags, dtype=np.float32)
        ft_dot = np.empty_like(end_flags, dtype=np.float32)
        for _ in range(max_iter):
            # 计算积分式的值（曲线的长度）（Gauss-Legendre积分）
            ft[~end_flags] = self.get_batch_line_length(curve_idx[~end_flags], start_t[~end_flags], end_t[~end_flags])
            # 积分式的导数计算，对积分式 L = int_t0^t sqrt(dx^2 + dy^2) dt 其中 t0 为定值，t 为待求参数
            # 上式即为变上限的积分式的微分，即 sqrt(dx^2 + dy^2) 在 t 处的值，所以直接使用被积函数代值即可
            ft_dot[~end_flags] = self._curve_integrand_func(support_points[~end_flags],
                                                            end_t[~end_flags][:, None],
                                                            curve_type=curve_type).squeeze(1)

            # 牛顿法迭代计算
            factor = (ft - length) / ft_dot
            end_t[~end_flags] = end_t[~end_flags] - factor[~end_flags]
            end_flags[~end_flags] = np.abs(factor[~end_flags]) < accuracy
            if np.all(end_flags):
                break

        if not np.all(end_flags):
            warnings.warn("Optimization not converged, some points may not be optimized")

        return self._batch_trace_points_from_support_points_and_t(support_points, end_t, curve_type=curve_type), end_t

    def get_batch_points_by_length(self, curve_idx, start_t, length, max_iter=10, accuracy=1e-6):
        """
        根据 curve_idx, start_t, length 得到对应曲线上一段的点
        (curve_idx和start_t用于确定曲线上一点的位置，length曲线长度，返回的点是以curve_idx和start_t确定的点往前长度为length的曲线上的点)

        :param curve_idx: 批量的 curve_idx，shape 为 (batch_size, requires_n_points)
        :param start_t: 批量的 start_t，shape 为 (batch_size, requires_n_points)
        :param length: 批量的 length，shape 为 (batch_size, requires_n_points)
        :param max_iter: 最大迭代次数
        :param accuracy: 精度

        :return: 批量的曲线上一段的点，shape 为 (batch_size, requires_n_points, 2)
        """
        assert curve_idx.shape == start_t.shape == length.shape, "curve_idx, start_t, length 必须有相同的 shape"

        end_points = np.zeros((*curve_idx.shape, 2))
        end_t = np.zeros_like(start_t)

        for curve_type in range(self.curve_type_num):
            # 选择 curve_type 对应的曲线，shape 为 (batch_size, requires_n_points)
            select_idx = self.point_type_idx[curve_idx] == curve_type
            if not np.any(select_idx):
                continue
            # 二维索引，经过选择后的 shape 为 (n(满足条件的元素个数),) (两个维度被压缩)
            if curve_type == 0:
                end_points[select_idx], end_t[select_idx] = self._get_points_by_length_p1(curve_idx[select_idx],
                                                                       start_t[select_idx], length[select_idx])
            elif curve_type == 1 or curve_type == 2:
                end_points[select_idx], end_t[select_idx] = self._get_points_by_length_p2_p3(curve_idx[select_idx],
                                                                          start_t[select_idx], length[select_idx],
                                                                          curve_type=curve_type, max_iter=max_iter,
                                                                          accuracy=accuracy)
            else:
                raise ValueError("curve_type 必须为 0, 1 或 2")

        return end_points, end_t

    def _get_rate_by_curve_idx_and_t_p1(self, curve_idx):
        support_points = np.stack(self.support_points[curve_idx]).astype(np.float32)
        rate = support_points[:, 1, :] - support_points[:, 0, :]
        rate_norm = rate / np.linalg.norm(rate, axis=1, keepdims=True)
        return rate_norm

    def _get_rate_by_curve_idx_and_t_p2_p3(self, curve_idx, t, curve_type=1):
        assert curve_type in [1, 2], "curve_type 必须为 1 或 2"

        support_points = np.stack(self.support_points[curve_idx]).transpose(0, 2, 1).astype(np.float32)
        if curve_type == 1:
            rate = support_points @ self.B2[None, :, :] @ self._ut2_diff(t[:, None])
        elif curve_type == 2:
            rate = support_points @ self.B3[None, :, :] @ self._ut3_diff(t[:, None])
        else:
            raise ValueError("curve_type 必须为 1 或 2")
        rate = rate.squeeze(-1)
        rate_norm = rate / np.linalg.norm(rate, axis=1, keepdims=True)
        return rate_norm

    def get_rate_by_curve_idx_and_t(self, curve_idx, t):
        """
        根据 curve_idx, t 得到对应曲线上一点的方向向量

        :param curve_idx: 批量的 curve_idx，shape 为 (batch_size, n)
        :param t: 批量的 t，shape 为 (batch_size, n)

        :return: 批量的曲线上一点的方向向量，shape 为 (batch_size, n, 2)
        """
        assert curve_idx.shape == t.shape, "curve_idx, t 必须有相同的 shape"

        rate_norm = np.zeros((*t.shape, 2))
        for type_idx in range(self.curve_type_num):
            select_idx = self.point_type_idx[curve_idx] == type_idx
            if not np.any(select_idx):
                continue

            if type_idx == 0:
                rate_norm[select_idx] = self._get_rate_by_curve_idx_and_t_p1(curve_idx[select_idx])
            elif type_idx == 1 or type_idx == 2:
                rate_norm[select_idx] = self._get_rate_by_curve_idx_and_t_p2_p3(curve_idx[select_idx],
                                                                                t[select_idx],
                                                                                curve_type=type_idx)
            else:
                raise ValueError("curve_type 必须为 0, 1 或 2")
        return rate_norm

    def _get_curvature_p2p3(self, curve_idx, t, curve_type=1):
        assert curve_type in [1, 2], "curve_type 必须为 1 或 2"
        support_points = np.stack(self.support_points[curve_idx]).transpose(0, 2, 1).astype(np.float32)
        if curve_type == 1:
            xy_dot = support_points @ self.B2[None, :, :] @ self._ut2_diff(t[:, None])
            xy_dot_dot = support_points @ self.B2[None, :, :] @ self._ut2_diff_diff(t[:, None])
        elif curve_type == 2:
            xy_dot = support_points @ self.B3[None, :, :] @ self._ut3_diff(t[:, None])
            xy_dot_dot = support_points @ self.B3[None, :, :] @ self._ut3_diff_diff(t[:, None])
        else:
            raise ValueError("curve_type 必须为 1 或 2")

        xy_dot = xy_dot.squeeze(-1)
        xy_dot_dot = xy_dot_dot.squeeze(-1)
        x_dot, y_dot = xy_dot[:, 0], xy_dot[:, 1]
        x_dot_dot, y_dot_dot = xy_dot_dot[:, 0], xy_dot_dot[:, 1]
        curvature = np.abs(x_dot * y_dot_dot - x_dot_dot * y_dot) / (x_dot ** 2 + y_dot ** 2) ** 1.5
        return curvature

    def get_curvature_by_curve_idx_and_t(self, curve_idx, t):
        """
        根据 curve_idx, t 得到对应曲线上一点的曲率

        :param curve_idx: 批量的 curve_idx，shape 为 (batch_size,)
        :param t: 批量的 t，shape 为 (batch_size,)

        :return: 批量的曲线上一点的曲率，shape 为 (batch_size,)
        """
        assert curve_idx.shape == t.shape, "curve_idx, t 必须有相同的 shape"

        curvature = np.zeros_like(t)
        for type_idx in range(self.curve_type_num):
            select_idx = self.point_type_idx[curve_idx] == type_idx
            if not np.any(select_idx):
                continue

            if type_idx == 0:
                curvature[select_idx] = 0
            else:
                curvature[select_idx] = self._get_curvature_p2p3(curve_idx[select_idx],
                                                                 t[select_idx], curve_type=type_idx)
        return curvature

    def show_trace(self, show_target_points=True, show_support_points=True, show=True, ax=None):
        assert self.curve is not None, "请先调用 generate_trace 方法生成轨迹"
        if not self.color_init:
            self.color_trace = plt.get_cmap('tab20')(np.linspace(0, 1, self.target_points.shape[0]))
            np.random.shuffle(self.color_trace)
            self.color_init = True

        if ax is None:
            if self.ax is not None:
                ax = self.ax
            else:
                _, self.ax = plt.subplots()
                ax = self.ax

        for i in range(self.curve.shape[0] - 1):
            ax.plot(self.curve[i, :, 0], self.curve[i, :, 1], color=self.color_trace[i], alpha=0.3)
            if show_support_points:
                ax.scatter(self.support_points[i][:, 0], self.support_points[i][:, 1], color=self.color_trace[i])
            if show_target_points:
                ax.scatter(self.target_points[i, 0], self.target_points[i, 1], color=self.color_trace[i])
                if i == self.curve.shape[0] - 1:
                    ax.scatter(self.target_points[-1, 0], self.target_points[-1, 1], color=self.color_trace[i])
        ax.axis("equal")

        if show:
            plt.show()

    @staticmethod
    def generate_random_colors(n):
        return np.random.rand(n, 3)

    def save_trace(self, file_path):
        assert self.curve is not None, "请先调用 generate_trace 方法生成轨迹"

        save_item = np.array([self.target_points, self.start_point_rate, self.end_point_rate,
                              self.curve, self.support_points, self.p1_points, self.p2_points, self.p3_points,
                              self.point_type_idx, self.curve_type_num, self.points_club],
                             dtype=object)
        np.save(file_path, save_item)

    def load_trace(self, file_path):
        load_item = np.load(file_path, allow_pickle=True)
        # 给定的初始点
        self.target_points = load_item[0]
        self.start_point_rate = load_item[1]
        self.end_point_rate = load_item[2]
        self.curve = load_item[3]
        # 给定的初始点加上插值点
        self.support_points = load_item[4]
        self.p1_points = load_item[5]
        self.p2_points = load_item[6]
        self.p3_points = load_item[7]
        self.point_type_idx = load_item[8]
        self.curve_type_num = load_item[9]
        self.points_club = load_item[10]


def sinesweep(f0, f1, sweeptime, samplingrate, peak):  # 扫频信号：起始频率f0、截止频率f1、采样率和幅度
    from sympy import symbols, diff, sin
    k = np.exp(np.log(float(f1) / f0) / sweeptime)  # 增长系数k的计算公式
    data_len = sweeptime * samplingrate  # 数据长度
    t = np.linspace(0, sweeptime, data_len)  # 起始时间
    p = 2 * np.pi * f0 / np.log(k)
    data = peak * np.sin(p * (np.power(k, t) - 1))  # 将每个采样点的幅度值存入数组
    x = symbols('x')
    y = peak * sin(p * (k ** x - 1))
    dy = diff(y, x)
    return t, data, dy.subs(x, t[0]), dy.subs(x, t[-1])


if __name__ == '__main__':

    # points = np.array([[0, 0],
    #                    [1, 1],
    #                    [2, 2],
    #                    [3, 3]])
    # r = 100
    # t = np.arange(0, np.pi * 3 / 2, 0.2)
    # x = np.cos(t) * r
    # y = np.sin(t) * r
    #
    # x1 = np.arange(0, r, 10)
    # y1 = x1 - r
    #
    # x = np.hstack((x, x1))
    # y = np.hstack((y, y1))
    #
    # print(x.shape, y.shape)
    #
    # points = np.vstack((x, y)).T
    # start_point_rate = np.array([-1, 0])
    # end_point_rate = np.array([0, 1])
    # tg = Trace(points, start_point_rate, end_point_rate)
    #
    # tg.generate_trace(mode="p2_p3")
    # # print(tg.support_points[0])
    # print(tg.point_type_idx)
    # tg.show_trace(show_support_points=True, show_target_points=False)
    # tg.save_trace(file_path="./trace/circle_line.npy")

    # points = np.array([[0, 0],
    #                    [10, 0],
    #                    [20, 10],
    #                    [30, 20],
    #                    [40, 10],
    #                    [50, 0],
    #                    [60, 0]]
    #                   )
    # start_point_rate = np.array([1, 0])
    # end_point_rate = np.array([1, 0])
    #
    # tg = Trace(points, start_point_rate, end_point_rate)
    # tg.generate_trace(mode="p2_p3")
    # print(tg.point_type_idx)
    # exit()
    #
    # tg.show_trace(show_support_points=True, show_target_points=False)
    # # tg.save_trace("trace/trace1.npy")

    # points = np.array([[0, 0],
    #                    [10, 0],
    #                    [20, 0],
    #                    [30, 0]], dtype=np.float32
    #                   )
    # start_point_rate = np.array([1, 0])
    # end_point_rate = np.array([1, 0])
    #
    # tg = Trace(points, start_point_rate, end_point_rate)
    # tg.generate_trace(mode="p2_p3")
    #
    # tg.show_trace(show_support_points=True, show_target_points=False)
    # tg.save_trace("trace/line.npy")

    SAMPLING_RATE = 100
    SWEEP_TIME = 100
    F0 = 0.01
    F1 = 0.05
    PEAK = 10
    # 扫频信号：起始频率f0、截止频率f1、采样率和幅度
    t, data, start_rate, end_rate = sinesweep(F0, F1, SWEEP_TIME, SAMPLING_RATE, PEAK)
    plt.title("sweep data")
    plt.plot(t, data)
    plt.show()

    xy = np.array([t[::200], data[::200]]).T
    xy[:, 0] += 10
    start_line = np.array([[0, 0],
                           [5, 0]])
    xy = np.vstack((start_line, xy))

    start_point_rate = np.array([1, 0], dtype=np.float32)
    end_point_rate = np.array([1, end_rate], dtype=np.float32)

    tg = Trace(xy, start_point_rate, end_point_rate)
    tg.generate_trace(mode="p2_p3")

    tg.show_trace(show_support_points=False, show_target_points=False)
    tg.save_trace("trace/sweep.npy")

    x = []
    curvature = []
    t_num = 100
    for curve_idx in range(tg.curve.shape[0] - 1):
        batch_t = np.linspace(0, 1, t_num)
        batch_curve_idx = np.ones_like(batch_t, dtype=int) * curve_idx
        batch_curvature = tg.get_curvature_by_curve_idx_and_t(batch_curve_idx, batch_t)
        xy = tg.get_trace_points_from_curveidx_and_t(batch_curve_idx, batch_t)

        x.append(xy[:, 0].copy())
        curvature.append(batch_curvature.copy())

    x = np.array(x).reshape(-1)[t_num:-t_num]
    curvature = np.array(curvature).reshape(-1)[t_num:-t_num]

    plt.plot(x, curvature)
    plt.show()