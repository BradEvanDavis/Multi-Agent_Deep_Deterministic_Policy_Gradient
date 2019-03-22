from unityagents import UnityEnvironment
import numpy as np
from agent import Agent
import torch
from collections import deque
import time
from tensorboardX import SummaryWriter
import vars

# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='Tennis.exe', no_graphics=False)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)

print('Number of agents:', num_agents)

action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
score = np.zeros(num_agents)
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def save_checkpoint(model, optimizer, save_path, episode_num,
                    min_scores, mean_scores, max_scores, moving_avgs,
                    scores_window, best_score, duration, scores):

    checkpointRes = {'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'episode_num': episode_num,
                     'mean_scores': mean_scores,
                     'min_scores': min_scores,
                     'max_scores': max_scores,
                     'moving_avgs': moving_avgs,
                     'scores_window': scores_window,
                     'best_score': best_score,
                     'duration': duration,
                     'scores': scores
                     }
    return torch.save(checkpointRes, save_path)


def train_loop(n_episodes=5000, conseq_games=40, train=True, num_agents=num_agents, print_every=20, train_mode=True,
         load=False, actor_loadFile=None, critic_loadFile=None):
    # --------------------------------------------------------------------------------------------------#
    writer = SummaryWriter('runs/MADDPG')
    solved = False
    counter = 0
    if load == True and counter == 0:

        actor_load = torch.load(actor_loadFile)
        critic_load = torch.load(critic_loadFile)
        stats = torch.load(critic_loadFile)

        scores = stats['scores']
        mean_scores = stats['mean_scores']
        min_scores = stats['min_scores']
        max_scores = stats['max_scores']
        best_score = stats['best_score']
        moving_avgs = stats['moving_avgs']
        duration = stats['duration']
        episode_start = stats['episode_num']
        scores_window = stats['scores_window']

        agent.actor_local.load_state_dict(actor_load['model_state_dict'])
        agent.actor_optimizer.load_state_dict(actor_load['optimizer_state_dict'])

        agent.critic_local.load_state_dict(critic_load['model_state_dict'])
        agent.critic_optimizer.load_state_dict(critic_load['optimizer_state_dict'])

    elif load == False and counter == 0:

        mean_scores = []
        min_scores = []
        max_scores = []
        best_score = -np.inf
        scores_window = deque(maxlen=100)
        moving_avgs = []
        duration = 0
        episode_start = 0
        scores = []

    # -----------------------------------------------------------------------------------------------------#
    time = 0.
    for episode in range(1, 5000 + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        episode_scores = []
        #start_time = time.time()
        counter += 1
        episode_num = episode_start + counter

        for time in range(1000):
            action = agent.act(state, add_noise=True)

            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations  # get the next state
            reward = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            agent.step(state, action, rewards, next_state, dones, game_num, i_episode)
            state = next_state
            episode_scores.append(reward)
            if np.any(dones):
                break

       # duration += time.time() - start_time
        min_scores.append(np.min(episode_scores))
        max_scores.append(np.max(episode_scores))
        mean_scores.append(np.mean(episode_scores))
        scores.append(sum(np.array(episode_scores).sum(1)))
        scores_window.append(scores[-1])
        moving_avgs.append(np.mean(scores_window))
        best_score = max(best_score, np.max(scores))

        scoresTB = scores[-1]
        MovingAVG_TB = moving_avgs[-1]
        best_scoreTB = best_score

        writer.add_scalar('Episode_Rewards', scoresTB, counter)
        writer.add_scalar('100_episode_MA', MovingAVG_TB, counter)
        writer.add_scalar('Best_Scores', best_scoreTB, counter)

        if episode_num % print_every == 0:
            print(
                '\rEpisode {}, Mean last 100: {:.3f}, Mean current: {:.3f}, Max: {:.3f}, Min: {:.3f}, Best_Score {:.3f}, Time: {:.2f}' \
                    .format(episode_num, moving_avgs[-1], mean_scores[-1],
                            max_scores[-1], min_scores[-1], best_score,
                            round(duration / 60, 2), end="\n"))

            for agent, num in zip(agents, range(2)):
                save_checkpoint(agent.actor_local, agent.actor_optimizer, 'checkpoint{}_actor.pth'.format(num),
                                episode_num=episode_num, min_scores=min_scores,
                                mean_scores=mean_scores, max_scores=max_scores,
                                moving_avgs=moving_avgs, scores_window=scores_window,
                                best_score=best_score, duration=duration, scores=scores)

                save_checkpoint(agent.critic_local, agent.critic_optimizer, 'checkpoint{}_critic.pth'.format(num),
                                episode_num=episode_num, min_scores=min_scores,
                                mean_scores=mean_scores, max_scores=max_scores,
                                moving_avgs=moving_avgs, scores_window=scores_window,
                                best_score=best_score, duration=duration, scores=scores)

        if moving_avgs[-1] >= 0.5 and solved == False:
            solved = True
            print('\n')
            print(
                '\nEnvironment solved in {:d} episodes with an 100 turn Moving Average Score of {:.2f} over 100 turns'.format(
                    episode_num, moving_avgs[-1]))
            print('\n')

            for agent, num in zip(agents, range(2)):
                save_checkpoint(agent.actor_local, agent.actor_optimizer, 'checkpoint{}_actor_best.pth'.format(num),
                                episode_num=episode_num, min_scores=min_scores,
                                mean_scores=mean_scores, max_scores=max_scores,
                                moving_avgs=moving_avgs, scores_window=scores_window,
                                best_score=best_score, duration=duration, scores=scores)

                save_checkpoint(agent.critic_local, agent.critic_optimizer, 'checkpoint{}_critic_best.pth'.format(num),
                                episode_num=episode_num, min_scores=min_scores,
                                mean_scores=mean_scores, max_scores=max_scores,
                                moving_avgs=moving_avgs, scores_window=scores_window,
                                best_score=best_score, duration=duration, scores=scores)

    writer.close()
    return mean_scores, moving_avgs, max_scores, min_scores


actor_loadFile = None
critic_loadFile = None

mean_scores, moving_avgs, max_scores, min_scores = train_loop(load=False, actor_loadFile=actor_loadFile,
                                                        critic_loadFile=critic_loadFile)
