import numpy as np
import random
import copy
from model import Actor, Critic
import torch
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
import torch.optim as optim
import vars


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_size, action_size, random_seed=44):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACT)

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        #replay memory + exploration via noise add
        self.noise = OUNoise(action_size, random_seed)
        self.buffer_size = vars.buffer_size
        self.gamma_start = vars.gamma_start
        self.gamma = vars.gamma
        self.epsilon = vars.epsilon
        self.epsilon_decay = vars.epsilon
        self.batch_size = vars.batch_size  # minibatch size
        self.TAU = vars.TAU # for soft update of target parameters
        self.LR_ACT = vars.LR_ACT  # learning rate of the actor
        self.LR_CRITIC = vars.LR_CRITIC  # learning rate of the critic
        self.WEIGHT_DECAY = vars.WEIGHT_DECAY  # L2 weight decay
        self.LEARN_NUM = vars.LEARN_NUM
        self.LEARN_EVERY = vars.LEARN_EVERY

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed)

    def step(self, state, action, reward, next_state, done, timestep, i_episode):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # learn
        if len(self.memory) > self.BATCH_SIZE and i_episode < 200:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA_START)
        elif len(self.memory) > self.BATCH_SIZE and timestep % self.LEARN_EVERY == 0:
            for _ in range(self.LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        # reutrn action based on current policy
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).detach().cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += (self.epsilon) * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        ''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        actor_target(state) -> action
        critic_target(state, action) -> q-value
        '''
        states, actions, rewards, next_states, dones = experiences

        # update critic -------------------------------------------
        self.critic_optimizer.zero_grad()
        self.actor.eval()
        index = torch.tensor([index]).to_device()
        # get predicted next state actions and Q values from targets
        actions_next = torch.cat(all_next_actions, dim=1).to(device)

        with torch.no_grad():
            Q_trgts_next = self.critic_target(next_states, actions_next)

        # compute Q targets for current state (y_i)
        Q_expected = self.critic_local(states, actions)
        Q_trgts = rewards + (self.GAMMA * Q_trgts_next * (1 - dones.index_select(1, index)))

        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(Q_expected, Q_trgts)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # update actor--------------------------------------------
        self.actor_optimizer.zero_grad()

        actions_pred = [actions if i == self.index else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)

        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update trgt network-------------------------------------

        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

        # noise updates--------------------------------------------
        self.epsilon -= self.EPSILON_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            TAU: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

class OUNoise:
    '''Ornstein-Uhlenbeck process'''

    def __init__(self, size, seed=44, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        '''reset noise to mean'''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''update internal state and return as noise'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    '''fixed buffer to store experiences'''

    def __init__(self, action_size, BUFFER_SIZE, BATCH_SIZE, seed=44):
        ''' BUFFER_SIZE = maximum size of buffer
            BATCH_SIZE = size of each training batch'''
        self.action_size = vars.action_size
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.BATCH_SIZE = vars.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''add experience to memory'''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''randomly sample a batch of experience from memory'''
        experiences = random.sample(self.memory, k=self.BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''current size of internal memory'''
        return len(self.memory)