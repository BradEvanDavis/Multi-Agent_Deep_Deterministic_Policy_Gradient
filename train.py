import numpy as np
import torch
import time
from tensorboardX import SummaryWriter
from collections import deque
from maddpg import maddpg
from unityagents import UnityEnvironment
from vars import Options


def save_checkpoint(model, optimizer, save_path, episode_num, moving_avgs, scores_window, duration):
    checkpointRes = {'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'episode_num': episode_num,
                     'moving_avgs': moving_avgs,
                     'scores_window': scores_window,
                     'duration': duration}
    torch.save(checkpointRes, save_path)


def train_loop(env, brain_name, agent, loader=False, actor_loadFile=None, critic_loadFile=None):

    counter = 0
    if loader == True and counter == 0:

        actor_load = torch.load(actor_loadFile)
        critic_load = torch.load(critic_loadFile)
        stats = torch.load(critic_loadFile)

        scores = stats['scores']
        avg_scores = stats['mean_scores']

        agent.actor_local.load_state_dict(actor_load['model_state_dict'])
        agent.actor_optimizer.load_state_dict(actor_load['optimizer_state_dict'])

        agent.critic_local.load_state_dict(critic_load['model_state_dict'])
        agent.critic_optimizer.load_state_dict(critic_load['optimizer_state_dict'])

    elif loader == False and counter == 0:

        scores = []
        scores_window = deque(maxlen=100)
        avg_scores = []
        writer = SummaryWriter('runs/run1')
        start_time = time.time()
        solved = False

    for e in range(1, Options.n_episodes + 1):

        rewards = []
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        counter += 1
        episode_num = e

        for t in range(Options.max_t):
            action = agent.act(state, add_noise=True)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            if any(done):
                break

        scores.append(sum(np.array(rewards).sum(1)))
        scores_window.append(scores[-1])
        avg_scores.append(np.mean(scores_window))
        duration = time.time() - start_time
        writer.add_scalar('stats/reward', scores[-1], e)
        writer.add_scalar('stats/avg_reward', avg_scores[-1], e)

        if e % Options.print_every == 0:
            print(f'E: {e:6} | Average: {avg_scores[-1]:10.4f} | Best Average: {max(avg_scores):10.4f} | Last Score: {scores[-1]:10.4f} | Elapsed Time: {round(time.time() - start_time,2)}', end='\r')
            for a, i in zip(maddpg.agents, [1,2]):
                save_checkpoint(a.actor_local, a.actor_optimizer, 'checkpoint_actor_local{}.pth'.format(i),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)

                save_checkpoint(a.critic_local, a.critic_optimizer, 'checkpoint_actor_local{}.pth'.format(i),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)

                save_checkpoint(a.actor_target, a.actor_optimizer, 'checkpoint_actor_target{}.pth'.format(i),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)

                save_checkpoint(a.critic_target, a.critic_optimizer, 'checkpoint_critic_target{}.pth'.format(i),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)

        if avg_scores[-1] >= 0.5 and solved == False:
            solved = True
            print('\nEnvironment solved in {:d} episodes with an 100 turn Moving Average Score of {:.2f} over 100 turns\n'.format(e, avg_scores[-1]))
            for a, i in zip(maddpg.agents, [1,2]):
                save_checkpoint(a.actor_local, a.actor_optimizer, 'checkpoint_actor_local{}.pth'.format(i+'_beat'),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)

                save_checkpoint(a.critic_local, a.critic_optimizer, 'checkpoint_actor_local{}.pth'.format(i+'_beat'),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)

                save_checkpoint(a.actor_target, a.actor_optimizer, 'checkpoint_actor_target{}.pth'.format(i+'_beat'),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)

                save_checkpoint(a.critic_target, a.critic_optimizer, 'checkpoint_critic_target{}.pth'.format(i+'_beat'),
                                episode_num=episode_num, moving_avgs=avg_scores, scores_window=scores_window,
                                duration=duration)


    writer.close()
    return scores, avg_scores

# -------------------------------------------------------------------------------------------------------------------

actor_loadFile = None
critic_loadFile = None

maddpg = maddpg()
env = UnityEnvironment(file_name='Tennis.exe', seed=Options.seed_num)
brain_name = env.brain_names[0]

train_loop(env=env, brain_name=brain_name, agent=maddpg, loader=False, actor_loadFile=actor_loadFile,
           critic_loadFile=critic_loadFile)
