from unityagents import UnityEnvironment
import random
import torch


class HyperParameters:
    def __init__(self):

        self.buffer_size = int(1e5)
        self.gamma_start = 0.9
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 1e-5
        self.batch_size = 256
        self.tau = 1e-3
        self.lr_act = 1e-3
        self.lr_critic = 1e-4
        self.weight_decay = 1e-8
        self.learn_num = 10
        self.learn_every = 20
        # Add OU Noise to actions
        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2
        self.noise_start = 1.0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Options:
    def __init__(self):

        self.n_episodes = 5000
        self.max_t = 1000
        self.print_every = 50
        self.update_frequency = 1
        self.seed_num = 47
        self.add_noise = True
        self.graphics = False
        self.option = True
        self.env_name = "Tennis.exe"


Options = Options()


class Environment:
    def __init__(self):

        self.num_agents = 2
        self.action_size = 2
        self.state_size = 24


HyperParameters = HyperParameters()
Environment = Environment()
