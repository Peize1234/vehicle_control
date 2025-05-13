
import numpy as np
import torch
from Model_Trace_Interactor import ModelTraceInteractor
from scipy.linalg import expm
from sympy import Matrix, symbols, exp, diag, integrate


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


def state_diff(real_state, reference_state):
    """
	Calculate the difference between two states in it's previous state. (mainly used for error encoder previous process)

	:param real_state: 车辆实际运行时的状态, shape (batch_size, seq_len + 1, state_dim)
	:param reference_state: 参考状态, shape (batch_size, seq_len + 1, state_dim)
	:return: 状态差, shape (batch_size, seq_len, state_dim)
	"""
    assert real_state.shape == reference_state.shape, "real_state and reference_state must have same shape"

    for i in range(1, real_state.shape[1]):
        transfer_real_tool = ModelTraceInteractor(real_state[:, i - 1, :])
        transfer_ref_tool = ModelTraceInteractor(reference_state[:, i - 1, :])

        real_state[:, i, :2] = transfer_real_tool.transfer_points_rates_to_vehicle_coordinate(
            real_state[:, i: i + 1, :2],
            None,
            site="center")[0].squeeze(1)
        reference_state[:, i, :2] = transfer_ref_tool.transfer_points_rates_to_vehicle_coordinate(
            reference_state[:, i: i + 1, :2],
            None,
            site="center")[0].squeeze(1)

    real_state[:, :2] = 0.0
    reference_state[:, :2] = 0.0

    real_state_diff = real_state[:, 1:, :] - real_state[:, :-1, :]
    reference_state_diff = reference_state[:, 1:, :] - reference_state[:, :-1, :]

    return real_state_diff - reference_state_diff


class MultiListContainer(object):
    def __init__(self, keys: list, num_sub_lists: int):
        self.dict_buffer = {}
        for key in keys:
            self.dict_buffer[key] = [[] for _ in range(num_sub_lists)]
        self.num_sub_lists = num_sub_lists

    def append(self, key: str, value: np.ndarray, done: np.ndarray = None):
        if done is None:
            assert value.shape[0] == self.num_sub_lists
            done = np.zeros(self.num_sub_lists, dtype=bool)
        else:
            assert done.shape[0] == self.num_sub_lists

        j = 0
        for i, d in enumerate(done):
            if not d:
                self.dict_buffer[key][i].append(value[j])
                j += 1

    def get(self, key):
        return self.dict_buffer[key]

    def set_item(self, key: str, value: np.ndarray, index: int = -1, done: np.ndarray = None):
        if done is None:
            assert value.shape[0] == self.num_sub_lists
            done = np.zeros(self.num_sub_lists, dtype=bool)
        else:
            assert done.shape[0] == self.num_sub_lists

        j = 0
        for i, d in enumerate(done):
            if not d:
                self.dict_buffer[key][i][index] = value[j]
                j += 1


def base_lengthways_control(now_speed, target_speed=3):
    """
    Get the action based on the current state using base lengthways control method.

    :return: The Fxf action for each vehicle. shape: (num_vehicles, action_space_dim)
    """
    return np.where(now_speed < target_speed, 4000, -4000)
