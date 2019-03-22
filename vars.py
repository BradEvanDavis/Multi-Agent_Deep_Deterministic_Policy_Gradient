
class hyperparameters(object):
    def __init__(self):
        self.buffer_size = 1e6
        self.gamma_start = 0.9
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 1e-7
        self.batch_size = 256  # minibatch size
        self.TAU = 1e-3  # for soft update of target parameters
        self.LR_ACT = 1e-3  # learning rate of the actor
        self.LR_CRITIC = 1e-4  # learning rate of the critic
        self.WEIGHT_DECAY = 1e-6  # L2 weight decay
        self.LEARN_NUM = 10
        self.LEARN_EVERY = 20

class training_params(object):
    def __init__(self):
        self.n_episodes = 5000
        self.max_t = 1000
        self.state_size = 24
        self.action_size = 2
        self.env_name = "Tennis.exe"
