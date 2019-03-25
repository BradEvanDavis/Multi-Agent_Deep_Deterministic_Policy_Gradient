from vars import HyperParameters as Params
from vars import Environment as Env
from vars import Options
from collections import namedtuple
from collections import deque
import numpy as np
import random
import copy
import torch


def save_checkpoint(model, optimizer, save_path, episode_num, mean_scores, moving_avg, scores_window, duration):
    checkpointRes = {'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'episode_num': episode_num,
                     'moving_avg': moving_avgs,
                     'scores_window': scores_window,
                     'duration': duration}

    torch.save(checkpointRes, save_path)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process"""
    def __init__(self):
        random.seed(Options.seed_num)
        np.random.seed(Options.seed_num)
        self.mu = Params.mu * np.ones(Env.action_size)
        self.theta = Params.theta
        self.sigma = Params.sigma
        self.action_size = Env.action_size
        self.reset()

    def reset(self):
        """reset noise to mean"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """update internal state and return as noise"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """
    Fixed buffer to store experiences
    BUFFER_SIZE = maximum size of buffer
    BATCH_SIZE = size of each training batch
    """
    def __init__(self):
        random.seed(Options.seed_num)
        np.random.seed(Options.seed_num)
        self.action_size = Env.action_size
        self.batch_size = Params.batch_size
        self.memory = deque(maxlen=Params.buffer_size)
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """add experience to memory"""
        return self.memory.append(self.experiences(state, action, reward, next_state, done))

    def sample(self):
        """randomly sample a batch of experience from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        done = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, done

    def __len__(self):
        """current size of internal memory"""
        return len(self.memory)


def critic_input(states, actions):
    return torch.cat((states, actions), dim=1)
