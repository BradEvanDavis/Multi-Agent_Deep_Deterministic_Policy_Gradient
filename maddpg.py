from vars import HyperParameters as Params
from vars import Environment as Env
from vars import Options
from utils import ReplayBuffer
from utils import OUNoise
import numpy as np
import torch
from ddpg import Ddpg
import random


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class maddpg():

    def __init__(self):
        random.seed(Options.seed_num)
        np.random.seed(Options.seed_num)

        self.buffer = ReplayBuffer()
        self.buffer_size = Params.buffer_size
        self.batch_size = Params.batch_size
        self.gamma = Params.gamma
        self.update_frequency = Options.update_frequency
        self.num_agents = Env.num_agents
        self.noise_start = Params.noise_start
        self.noise_decay = Params.epsilon_decay
        self.timestep = 0

        self.agents = [Ddpg(index) for index in range(self.num_agents)]

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)
        all_next_states = all_next_states.reshape(1, -1)
        self.buffer.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        self.timestep = (self.timestep + 1) % self.update_frequency
        if self.timestep == 0 and (len(self.buffer) > self.batch_size):
            experiences = [self.buffer.sample() for _ in range(self.num_agents)]
            self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, add_noise)
            #self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)

    def learn(self, experiences, gamma):
        all_actions = []
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            all_actions.append(agent.actor_local(state))
            all_next_actions.append(agent.actor_target(next_state))
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)

