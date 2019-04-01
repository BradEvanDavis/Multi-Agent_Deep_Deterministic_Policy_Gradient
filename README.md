# MultiActor_DeepDeterminnisticPolicyGradient
Two Agents acting on each other in a game of tennis

# Project Background
In the real world it typically takes multiple people to accomplish a certain task, likewise AI systems in the future (and the present) will need to be able to work cooperatively with other agents, including AIs and their human counterparts.  This project demonstrates a small step towards that end goal by placing multiple instances of a single agent into an environment where they must compete with one another.

Throughout this project, former work done in the deep deterministic policy gradient space was highly leveraged and served as the primary backbone of this code (see ddpg.py).  Leveraging the ddpg framework, the goal of this particular project was to keep the ball live as long as possible and scoring in this environment reflected that overall goal.  

# Goals
Project scores were determined based on consecutive hits by each agent.   Once an agent successfully returned the ping pong ball the agent would then receive an award of +0.1 points.  However, if an agent missed the ball / hit the ball out of bounds it would then receive -0.01 points instead.  Final scores for each episode was then calculated based on the maximum score by a single agent. The model was trained for a total of 5,000 episode where it achieved a benchmark score 100 episode moving averaging +0.5 pts after 2,337 episodes.

<img>'tensorboard_screenshot.PNG'</img>

