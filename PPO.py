import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import itertools
import pickle
from pathlib import Path
import os


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self, num_vehicles=1, buffer_path: Path = Path(r"./buffer"), buffer_length: int = 8000):
        self.num_vehicles = num_vehicles
        self.buffer_path = buffer_path
        self.buffer_name = "buffer"
        self.buffer_length = buffer_length

        self.init_buffer()

    def init_buffer(self):
        self.actions = [[] for _ in range(self.num_vehicles)]
        self.states = [[] for _ in range(self.num_vehicles)]
        self.logprobs = [[] for _ in range(self.num_vehicles)]
        self.rewards = [[] for _ in range(self.num_vehicles)]
        self.state_values = [[] for _ in range(self.num_vehicles)]
        self.is_terminals = [[] for _ in range(self.num_vehicles)]
        self.u = [[] for _ in range(self.num_vehicles)]

        self.total_lists = [self.actions, self.states, self.logprobs, self.rewards,
                            self.state_values, self.is_terminals, self.u]
        self.name2list = {"actions": self.actions, "states": self.states, "logprobs": self.logprobs,
                          "rewards": self.rewards,
                          "state_values": self.state_values, "is_terminals": self.is_terminals, "u": self.u}

    def clear(self):
        self.init_buffer()
        self.clear_buffer_files()

    def clear_buffer_files(self):
        for file_name in os.listdir(self.buffer_path):
            os.remove(self.buffer_path / file_name)

    def format(self):
        self.load_buffer()

        self.actions = list(itertools.chain.from_iterable(self.actions))
        self.states = list(itertools.chain.from_iterable(self.states))
        self.logprobs = list(itertools.chain.from_iterable(self.logprobs))
        self.rewards = list(itertools.chain.from_iterable(self.rewards))
        self.state_values = list(itertools.chain.from_iterable(self.state_values))
        self.is_terminals = list(itertools.chain.from_iterable(self.is_terminals))
        self.u = list(itertools.chain.from_iterable(self.u))

    def save_buffer(self):
        buffer_num = len(os.listdir(self.buffer_path))
        pickle.dump(self.total_lists, open(self.buffer_path / f"{self.buffer_name}{buffer_num}.pkl", "wb"))

    def load_buffer(self):
        temp_actions = [[] for _ in range(self.num_vehicles)]
        temp_states = [[] for _ in range(self.num_vehicles)]
        temp_logprobs = [[] for _ in range(self.num_vehicles)]
        temp_rewards = [[] for _ in range(self.num_vehicles)]
        temp_state_values = [[] for _ in range(self.num_vehicles)]
        temp_is_terminals = [[] for _ in range(self.num_vehicles)]
        temp_u = [[] for _ in range(self.num_vehicles)]

        for idx in range(len(os.listdir(self.buffer_path))):
            buffer_file = self.buffer_path / f"{self.buffer_name}{idx}.pkl"
            temp_lists = pickle.load(open(buffer_file, "rb"))
            for i in range(self.num_vehicles):
                temp_actions[i].extend(temp_lists[0][i])
                temp_states[i].extend(temp_lists[1][i])
                temp_logprobs[i].extend(temp_lists[2][i])
                temp_rewards[i].extend(temp_lists[3][i])
                temp_state_values[i].extend(temp_lists[4][i])
                temp_is_terminals[i].extend(temp_lists[5][i])
                temp_u[i].extend(temp_lists[6][i])

        for i in range(self.num_vehicles):
            temp_actions[i].extend(self.actions[i])
            temp_states[i].extend(self.states[i])
            temp_logprobs[i].extend(self.logprobs[i])
            temp_rewards[i].extend(self.rewards[i])
            temp_state_values[i].extend(self.state_values[i])
            temp_is_terminals[i].extend(self.is_terminals[i])
            temp_u[i].extend(self.u[i])

        self.actions = temp_actions
        self.states = temp_states
        self.logprobs = temp_logprobs
        self.rewards = temp_rewards
        self.state_values = temp_state_values
        self.is_terminals = temp_is_terminals
        self.u = temp_u

        self.total_lists = [self.actions, self.states, self.logprobs, self.rewards,
                            self.state_values, self.is_terminals, self.u]
        self.name2list = {"actions": self.actions, "states": self.states, "logprobs": self.logprobs, "rewards": self.rewards,
                          "state_values": self.state_values, "is_terminals": self.is_terminals, "u": self.u}

    def check_buffer_size(self) -> bool:
        total_size = 0
        for i in range(self.num_vehicles):
            total_size += len(self.actions[i])
            if total_size >= self.buffer_length:
                return True
        return False

    def append(self, keys: list, selected_values: list, done_idx: np.ndarray) -> None:
        """
        Append selected values to corresponding lists in the buffer.

        :param keys: list of keys to select values from
        :param selected_values: list of values to append (this value should be indexed by done_idx)
        :param done_idx: boolean array indicating which vehicles have terminated
        """
        assert len(keys) == len(selected_values), "Number of keys and values should be equal"
        assert np.sum(done_idx) == len(keys), "Number of True indices and keys should be equal"
        assert len(done_idx) == self.num_vehicles, "Number of done indices should be equal to number of vehicles"

        j = 0
        for i in range(len(done_idx)):
            if not done_idx[i]:
                for key, value in zip(keys, selected_values):
                    assert key in self.name2list, f"Key {key} not in buffer"
                    self.name2list[key][i].append(value[j])
                j += 1

        # Save buffer if buffer size is reached
        # 防止内存溢出（内存溢出会导致使用虚拟内存，导致系统卡顿，并且程序执行缓慢），每当buffer满了就保存一次
        if self.check_buffer_size():
            self.save_buffer()
            self.init_buffer()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        elif not isinstance(state, torch.Tensor):
            raise TypeError("state should be numpy array or torch tensor")

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, num_vehicles=1):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer(num_vehicles)

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state: np.ndarray, done: np.ndarray) -> np.ndarray:
        """
        Select action for given state as per current policy.
        :param state: state to select action for
        :param done: (optional) done flag to indicate that episode has terminated
        :return: action to take (numpy array)
        """
        assert state.shape[0] == np.sum(~done), "Number of states and done flags should be equal"

        if self.has_continuous_action_space:
            with torch.no_grad():
                action, action_logprob, state_val = self.policy_old.act(state)

            state = torch.FloatTensor(state).to(device)

            # j = 0
            # for i in range(len(done)):
            #     if not done[i]:
            #         self.buffer.states[i].append(state[j])
            #         self.buffer.actions[i].append(action[j])
            #         self.buffer.logprobs[i].append(action_logprob[j])
            #         self.buffer.state_values[i].append(state_val[j])
            #         j += 1
            self.buffer.append(["states", "actions", "logprobs", "state_values"],
                               [state, action, action_logprob, state_val],
                               done)

            return action.detach().cpu().numpy()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.extend([s for s in state])
            self.buffer.actions.extend([a for a in action])
            self.buffer.logprobs.extend([al for al in action_logprob])
            self.buffer.state_values.extend([sv for sv in state_val])

            return action.item()

    def update(self):

        self.buffer.format()
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_u = torch.stack(self.buffer.u, dim=0).detach().to(device)  # (batch_size, 1)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            if hasattr(self.policy, "fuser"):
                logprobs, state_values, dist_entropy, error_encoder_output = self.policy.evaluate(old_states.cpu().numpy(), old_actions)
            else:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # if hasattr(self.policy, "fuser"):
            #     loss += 1.0 * self.MseLoss(error_encoder_output, old_u.float())
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


