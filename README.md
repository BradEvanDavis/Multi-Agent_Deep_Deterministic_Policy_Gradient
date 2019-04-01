# MultiActor_DeepDeterminnisticPolicyGradient
Two Agents acting on each other in a game of tennis

# Project Background
In the real world it typically takes multiple people to accomplish a certain task, likewise AI systems in the future (and the present) will need to be able to work cooperatively with other agents, including AIs and their human counterparts.  This project demonstrates a small step towards that end goal by placing multiple instances of a single agent into an environment where they must compete with one another.

Throughout this project, former work done in the deep deterministic policy gradient space was highly leveraged and served as the primary backbone of this code (see ddpg.py).  Leveraging the ddpg framework, the goal of this particular project was to keep the ball live as long as possible and scoring in this environment reflected that overall goal.  

# Goals
Project scores were determined based on consecutive hits by each agent.   Once an agent successfully returned the ping pong ball the agent would then receive an award of +0.1 points.  However, if an agent missed the ball / hit the ball out of bounds it would then receive -0.01 points instead.  Final scores for each episode was then calculated based on the maximum score by a single agent. The model was trained for a total of 5,000 episode where it achieved a benchmark score 100 episode moving averaging +0.5 pts after 2,337 episodes.


![Results](https://github.com/BradEvanDavis/MultiActor_DeepDeterminnisticPolicyGradient/raw/master/tensorboard_screenshot.PNG)

# Multi Agent Deep Deterministic Policy Gradient Explained:
This actor-critic implementation utilizes deep reinforcement learning known as Deep Deterministic Policy Gradient (DDPG) to evaluate a continuous action space. DDPG is based on the papers ‘Deterministic Policy Gradient Algorithms’ published in 2014 by David Silver and ‘Continuous Control with Deep Reinforcement Learning’ published by Tomothy P. Lillicrap in 2015.  To continue exploration of DDPG, and inspired by “Multi Agent Actor Critic for Mixed Cooperative Competitive environments “ by OpenAI, the functionality of the original DDPG algorithm was able to be enhanced by creating multiple instances of the same agent within the 'Tennis' environment allowing the agents to learn to competitively cooperate with one-another while playing a game of table tennis.

Unlike other actor-critic methods that rely on stochastic distributions to return probabilities across a discreet action space, DDPG utilizes a deterministic policy to directly estimate a set of continuous actions based on the environment’s current state. As a result, DDPG is able to take advantage of Q values (much like DQN) which allows the estimation of rewards by maximizing Q via a feed-forward Critic network. The actor feed-forward network then is able to use the critic’s value estimates to choose the action that maximizes Q via back-propagation (stochastic gradient decent of the deterministic policy gradient allows optimizing for Q by minimizing MSE).

Like DQN, and DDPG, Multi Agent DDPG requires the agents to explore the environment in-order to determine an optimal policy – this is accomplished by adding noise via the Ornstein-Uhlenbeck process (ON) to explore the environment. However, unlike typical DDPG algorithms, learning is completed across multiple agents competing against one-another within the tennis environment across a single replaybuffer.

Interestingly, after maxing out after 2.5K episodes with a 100 episode MA score at 1.35, the model then varied lower, but after 1.4K additional episodes (3.9K)  it reached a new high of 1.406 displaying an upward trending score for the remaining episodes and then finally peaking  at 1.55 after 4,670 episodes.

Next steps include implementing a prioritized replay buffer or implementing a d4pg agent instead of ddpg.  Furthermore, it would be interesting to test whether or not the current model would continue to improve with additional episodes post-5,000  continuing the upward trend demonstrated in the first 5,000 episodes.

# How to run:
1.  Clone this repository onto your machine and unzip into a directory of your choice
2.  Download and install Anaconda if needed
3.  Create a new environment and install the package requirements listed below
4.  Download the Unity environment and unzip the executable into the project’s parent directory. 
  a.	It is recommended that your video card be CUDA compatible for optimal performance
5.  Unzip and load the trained models for actors and critics  

# Requirements:
1.  python 3.6+
2.  pytorch 1.0+ (Instructions: https://pytorch.org/)
3.  CUDA 9.0+ (optional)
4.  UnityAgent (Instructions: https://github.com/Unity-Technologies/ml-agents)
5.  Numpy
6.  Matplotlib
7.  Tennis Unity Environment (Linux),(OSX), (Win64),(Win32)
8.  TensorBoard

# Running the Code:
1.  Train the agent by running the appropriate training loop within train.py until obtaining a 100 episode moving average >= 0.5 pts.
2.  Refer to TensorBoard outputs by calling 'tensorboard --logdir runs --host localhost'
3.  Watch the actions taken by your newly trained smart agents by loading saved checkpoints into the training loop and setting the train variable to False in the vars.py file
  a.  Note that all hyperparameters can be modified through vars.py
