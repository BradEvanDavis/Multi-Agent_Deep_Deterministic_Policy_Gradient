import numpy as np
import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from vars import HyperParameters as Params
from vars import Options
from vars import Environment as env
from utils import OUNoise
import random
import utils

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class Ddpg:
    def __init__(self, index):
        random.seed(Options.seed_num)
        np.random.seed(Options.seed_num)

        self.noise = OUNoise()
        self.index = index
        self.action_size = env.action_size
        self.state_size = env.state_size
        self.tau = Params.tau
        self.epsilon = Params.epsilon

        self.actor_local = model.Actor(env.state_size, env.action_size)
        self.actor_target = model.Actor(env.state_size, env.action_size)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=Params.lr_act)

        self.critic_local = model.Critic(env.state_size, env.action_size)
        self.critic_target = model.Critic(env.state_size, env.action_size)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=Params.lr_critic, weight_decay=Params.weight_decay)

    def act(self, state, add_noise):
        # return action based on current policy
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(torch.from_numpy(state).float().to(device)).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += (self.epsilon * self.noise.sample())
            self.epsilon -= self.epsilon * Params.epsilon_decay
            self.epsilon = max(0, self.epsilon)
        return np.clip(action, -1, 1)

    def learn(self, index, experiences, gamma, all_next_actions, all_actions):
        states, actions, rewards, next_states, done = experiences

        # update critic -------------------------------------------
        self.critic_optimizer.zero_grad()

        index = torch.tensor([index]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)

        with torch.no_grad():
            Q_trgts_next = self.critic_target(utils.critic_input(next_states, actions_next))
        Q_expected = self.critic_local(utils.critic_input(states, actions))
        Q_trgts = rewards.index_select(1, index) + (gamma * Q_trgts_next * (1 - done.index_select(1, index)))
        F.mse_loss(Q_expected, Q_trgts.detach()).backward()

        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()

        # update actor--------------------------------------------

        actions_pred = [actions if i == self.index else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(utils.critic_input(states, actions_pred)).mean()
        actor_loss.backward()

        self.actor_optimizer.step()

        # update trgt network-------------------------------------
        self.actor_target.soft_update(self.actor_local, self.tau)
        self.critic_target.soft_update(self.critic_local, self.tau)

        # # noise updates--------------------------------------------
        # self.epsilon -= Params.epsilon_decay
        # OUNoise.reset()


