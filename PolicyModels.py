import numpy as np
import torch
from torch import nn

from env import SimEnv
from ErrorEncoder import ErrorEncoder
from MotionModel import BicycleModel, state_space_aug_dim, state_space_dim, u_ref
from ClassicalController import ClassicalController
from utils import state_diff
from PPO import *
import math
from LinearTrackingAdaptiveController import LinearTrackingAdaptiveController
from typing import Union
from copy import deepcopy


class StateEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.1):
        super(StateEncoder, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.active = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.x_encoder = nn.Linear(hidden_dim, output_dim)
        self.q_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.k_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.v_encoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x_ = self.embedding(x)
        x = self.active(x_)
        x = self.norm(x + x_)

        x = self.x_encoder(x)
        q = self.q_encoder(x)
        k = self.k_encoder(x)
        v = self.v_encoder(x)
        return q, k, v, x


class Fuser(nn.Module):
    def __init__(self, state_input_dim, fuser_output_dim, hidden_dim=128, num_head=8,
                 dropout=0.1):
        super(Fuser, self).__init__()

        self.state_est_encoder = StateEncoder(state_input_dim, hidden_dim, hidden_dim, dropout)
        self.state_now_encoder = StateEncoder(state_input_dim, hidden_dim, hidden_dim, dropout)
        self.output_head_controller = nn.Sequential(
            nn.Linear(hidden_dim, fuser_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fuser_output_dim),
        )
        self.output_head_state = nn.Sequential(
            nn.Linear(hidden_dim, fuser_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fuser_output_dim),
        )
        self.relu = nn.ReLU()

        self.num_head = num_head

    def forward(self, state_est, state_now, error_encoder_output):
        # (batch_size, hidden_dim)
        qc, kc, vc, xc = self.state_est_encoder(state_est)
        qs, ks, vs, xs = self.state_now_encoder(state_now)
        batch_size, hidden_dim = xc.size()

        qc, kc, vc, xc = self.split([qc, kc, vc, xc])
        qs, ks, vs, xs = self.split([qs, ks, vs, xs])

        qc = self.relu(qc)
        kc = self.relu(kc)
        qs = self.relu(qs)
        ks = self.relu(ks)

        # qc = qc / torch.linalg.norm(qc, dim=-1, keepdim=True)
        # kc = kc / torch.linalg.norm(kc, dim=-1, keepdim=True)
        # qs = qs / torch.linalg.norm(qs, dim=-1, keepdim=True)
        # ks = ks / torch.linalg.norm(ks, dim=-1, keepdim=True)

        qcks = torch.sum(qc * ks, dim=-1, keepdim=True) / math.sqrt(hidden_dim // self.num_head)
        qskc = torch.sum(qs * kc, dim=-1, keepdim=True) / math.sqrt(hidden_dim // self.num_head)

        xc_aug = xc + qskc * vs
        xs_aug = xs + qcks * vc

        xc_aug = xc_aug.view(batch_size, hidden_dim)
        xs_aug = xs_aug.view(batch_size, hidden_dim)

        xc_aug = self.output_head_controller(xc_aug)
        xs_aug = self.output_head_state(xs_aug)

        return torch.cat([xc_aug, xs_aug, error_encoder_output], dim=-1)

    def split(self, x_list):
        x = torch.stack(x_list, dim=0)
        num_x, batch_size, hidden_dim = x.size()
        assert hidden_dim % self.num_head == 0

        x = x.view(num_x, batch_size, self.num_head, hidden_dim // self.num_head)
        return x


class ActorCriticPlus(ActorCritic):
    def __init__(self, policy_input_dim, action_dim, has_continuous_action_space, action_std_init,
                 Classical_Controller: Union[ClassicalController, LinearTrackingAdaptiveController],
                 State_Action_Processor: SimEnv,
                 Error_Encoder: ErrorEncoder,
                 Fuser_Model: Fuser):
        super(ActorCriticPlus, self).__init__(policy_input_dim, action_dim, has_continuous_action_space,
                                              action_std_init)
        self.classical_controller = Classical_Controller
        self.state_action_processor = State_Action_Processor
        self.error_encoder = Error_Encoder
        self.fuser = Fuser_Model

    def act(self, data_input):
        """
        :param data_input: state space, target points and actions in global frame,
                           shape (batch_size, seq_len + 1, state_dim(aug) + 2 * target_points_num + action_dim)
        """
        _, policy_input = organize_policy_input(self.error_encoder,
                                                self.fuser,
                                                self.state_action_processor,
                                                data_input)

        return super(ActorCriticPlus, self).act(policy_input)

    def evaluate(self, data_input, action):
        error_encoder_output, policy_input = organize_policy_input(self.error_encoder,
                                                                   self.fuser,
                                                                   self.state_action_processor,
                                                                   data_input)

        return *super(ActorCriticPlus, self).evaluate(policy_input, action), error_encoder_output


class PPOPlus(PPO):
    def __init__(self, policy_input_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space,
                 Classical_Controller: Union[ClassicalController, LinearTrackingAdaptiveController],
                 State_Action_Processor: SimEnv,
                 Error_Encoder: ErrorEncoder,
                 Fuser_Model: Fuser,
                 action_std_init: float = 0.6,
                 num_vehicles: int = 1):
        super(PPOPlus, self).__init__(policy_input_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                                      has_continuous_action_space, action_std_init, num_vehicles)

        self.policy = ActorCriticPlus(policy_input_dim, action_dim, has_continuous_action_space, action_std_init,
                                      Classical_Controller,
                                      State_Action_Processor,
                                      Error_Encoder,
                                      Fuser_Model).to(device)

        self.policy_old = ActorCriticPlus(policy_input_dim, action_dim, has_continuous_action_space, action_std_init,
                                          Classical_Controller,
                                          State_Action_Processor,
                                          Error_Encoder,
                                          Fuser_Model).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())


def organize_policy_input(error_encoder_model: ErrorEncoder,
                          fuser_model: Fuser,
                          state_action_processor: SimEnv,
                          data_arr: np.ndarray,
                          action_dim: int = 2):
    """
    step function for inference

    :param error_encoder_model: error encoder model
    :param fuser_model: fuser model
    :param state_action_processor: base motion model、 model_trace interactor and data formatter
    :param data_arr: real state space, target points (result of env.format_state) and actions in global frame,
                     shape (batch_size (num_not_done(for data generation) or total_batch(for training)), seq_len + 1, \
                            state_dim(aug) + 2 * trace_points_num(transferred to vehicle frame) + action_dim)
                     note: action in  last sequence  is just the absolute control signal(classical controller steering angle and zero Fxf)
                           but others are full control signal (classical plus nn control signal)
    :param action_dim: action dimension

    :return:
    """
    # (batch size, seq len + 1, n)
    batch_size, seq_len_plus1, n = data_arr.shape
    state_aug_seq = data_arr[:, :, :state_space_aug_dim].copy()
    state_seq = data_arr[:, :, :state_space_dim].copy()
    trace_points_last = data_arr[:, -1, state_space_aug_dim:-action_dim].copy()  # (batch size, 2 * trace points num)
    action_normalized_seq = data_arr[:, :, -action_dim:].copy()

    last_normalized_acton = action_normalized_seq[:, -1, :].copy()  # (batch size, action dim)
    last_normalized_acton[:, 1] = np.ones_like(action_normalized_seq[:, -2, 1]) * 2000  # set Fxf to 2000 (used to estimate next state)
    # last_normalized_acton[:, 1] = action_normalized_seq[:, -2, 1]  # set Fxf to before action's Fxf (used to estimate next state)

    assert np.all(-1 <= action_normalized_seq[:, :, 0]) and np.all(action_normalized_seq[:, :, 0] <= 1) and \
           np.all(-1 <= action_normalized_seq[:, :, 1]) and np.all(action_normalized_seq[:, :, 1] <= 1), \
           "Action space is not valid"

    # (batch size, action dim)
    # classical_control_signal = torch.from_numpy(action_normalized_seq[:, -1, 0:1]).to(device)

    # (batch size, state aug dim)
    state_action_processor.set_state_space(state_aug_seq[:, 0, :])
    # (batch size, seq len + 1, state dim)
    # step_n 函数在类 BicycleModel 中定义，此处不涉及 env 中与路径相关的互交
    simulate_state = state_action_processor.step_n(action_normalized_seq[:, :-1, 0],
                                                   action_normalized_seq[:, :-1, 1],
                                                   np.ones((batch_size, seq_len_plus1 - 1),
                                                           dtype=np.float32) * u_ref,
                                                   return_aug_space=False,
                                                   add_raw_state_space=True,
                                                   used_normalized_action=True)

    # (batch size, seq len, state dim)
    real_sim_diff = state_diff(state_seq, simulate_state) * 1e2

    error_encoder_input = torch.from_numpy(
        np.concatenate((real_sim_diff, action_normalized_seq[:, :-1, :]), axis=-1)
    ).to(device)

    # (batch size, 1), (batch size, path state dim)
    u_est, path_state_estimate = error_encoder_model(error_encoder_input.float())

    # 最后一时刻的状态是当前的状态，利用最后一时刻的状态估计下一时刻的状态
    state_action_processor.set_state_space(state_aug_seq[:, -1, :])
    state_est = state_action_processor.step(last_normalized_acton, normalized=True)[0]
    state_est = torch.from_numpy(np.hstack((state_est[:, 2:state_space_dim],
                                            state_est[:, state_space_aug_dim:]))).to(device)

    # build state input as same as train stage 1 (batch size, (state dim - 2) + (2 * trace points num))
    state_now = torch.from_numpy(np.hstack((state_seq[:, -1, 2:], trace_points_last))).to(device)

    state_est[:, :state_space_dim - 2] /= torch.tensor([[state_action_processor.phi_limit[1],
                                                         state_action_processor.Ux_limit[1],
                                                         state_action_processor.Uy_limit[1],
                                                         state_action_processor.r_limit[1]]], device=device)
    state_now[:, :state_space_dim - 2] /= torch.tensor([[state_action_processor.phi_limit[1],
                                                         state_action_processor.Ux_limit[1],
                                                         state_action_processor.Uy_limit[1],
                                                         state_action_processor.r_limit[1]]], device=device)

    fuser_input = (state_est.float(), state_now.float(), path_state_estimate)
    fuser_output = fuser_model(*fuser_input)

    return u_est, fuser_output


if __name__ == '__main__':
    model = Fuser(state_input_dim=6, fuser_output_dim=10, hidden_dim=128, dropout=0.1)

    x1 = np.random.rand(100, 1)
    x2 = np.random.rand(100, 6)
    x3 = np.random.rand(100, 10)

    print(model(torch.tensor(x1).float(), torch.tensor(x2).float(), torch.tensor(x3).float()).shape)
