import numpy as np
import torch
from agent import DDPGAgent
import vars

class MADDPGAgent():
    def __init__ (self):

        self.buffer_size = vars.hyperparameters.buffer_size
        self.batch_size = vars.hyperparameters.batch_size
        self.update_freq = vars.hyperparameters.update_freq
        self.gamma = vars.hyperpparameters.gamma
        self.epsilon = vars.hyperparameters.epsilon
        self.memory = DDPGAgent.ReplayBuffer(vars.memory)
        self.num_agents = vars.env.num_agents
        self.time = 0

        self.agents = [DDPGAgemt(index, vars) for index in range self.num_agents]

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1,-1)
        all_next_states = all_next_states.reshape(1, -1)
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)

        self.time = (self.time + 1) % self.update_freq
        if self.time == 0 and (len(self.memory)) > self.batch_size):
            experiences = [self.memory.sample() for _ in range(self.num_agents)]
            self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise=True)
            self.noise *= self.epsilon
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)

    def learn(self, experiences, gamma):
        all_actions = []
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state=states.reshape[-1,2,24].index_select(1, agent_id).squeeze(1)
            all_actions.append(agent.actor_local(state))
            all_next_actions.append(agent.actor_target(next_state))

        for i, agent in enumerate(sekf.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)



