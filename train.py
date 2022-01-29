import glob
import os
import sys
import random
import time
import numpy as np
import pandas as pd
import cv2
import math
import tensorflow as tf
import tensorflow.keras.backend as backend
from collections import deque
from threading import Thread

from tqdm import tqdm

import settings
from sources import agent
from sources import carla


def open_carla(require):
    try:
        if require == 'fast':
            os.popen(
                'D:\\Program\\CarlaSimulator_v0.9.12\\CarlaUE4.exe -benchmark  -fps=10 -quality-level=Low')
        else:
            os.popen('D:\\Program\\CarlaSimulator_v0.9.12\\CarlaUE4.exe')
    except Exception as err:
        print(err)
    print('opening Carla')


def close_carla():
    try:
        os.system('TASKKILL /F /IM CarlaUE4.exe')
    except Exception as err:
        print(err)
    time.sleep(0.5)


def carla_is_running():
    import psutil
    if "CarlaUE4.exe" in (p.name() for p in psutil.process_iter()):
        return True


def reset_world():
    Env.world.wait_for_tick()
    for x in list(Env.world.get_actors()):
        if x.type_id == 'vehicle.tesla.model3' or x.type_id == 'sensor.lidar.ray_cast' or x.type_id == 'sensor.other.collision':
            x.destroy()


def save_model(name, n, df, agent):
    file_path = "DATA\\"
    df.to_csv(file_path+'{}_{}.csv'.format(name, n))
    agent.save_critic(file_path+'critic'+name+'_'+str(n))
    agent.save_actor(file_path+'actor'+name+'_'+str(n))


if __name__ == '__main__':

    if carla_is_running():
        pass
    else:
        close_carla()
        open_carla('fast')
        time.sleep(17)

    FPS = 60
    ep_rewards = []
    ep = []
    avg = 0
    avg_reward = []
    Step = []
    Explore = []
    Steer = []
    Epsilon = []

    random.seed(1)
    np.random.seed(1)
    # tf.random.set_seed(1)

    # Create models folder
    if not os.path.isdir('DATA'):
        os.makedirs('DATA')

    # sleepy = 0.3

    #pp = ProgressPlot(x_label="Episode",line_names=['Average_reward'])

    # if LOAD == True:
    #     # In case Train from loaded_model
    #     Agent = DDPG_load_model(loaded_critic, loaded_actor)
    #     epsilon = load_epsilon
    #     nn = 0
    #     for i in df_load.Reward:
    #         avg = ((avg*(nn)+i)/(nn+1))
    #         # pp.update(float(avg))
    #         nn += 1
    #     avg = sum(df_load.Reward)/df_load.shape[0]
    # else:

    #     # Create Agent and environment
    #     Agent = agent.DDPGAgent()
    #     load_episode = 0

    Agent = agent.DDPGAgent()
    load_episode = 0
    # Iterate over episodes
    Env = carla.CarEnv()

    # reset_world()

    # Start training thread and wait for training to be initialized
    # trainer_thread = Thread(target=Agent.train_in_loop, daemon=True)
    # trainer_thread.start()

    # while not Agent.training_initialized:
    #     time.sleep(0.01)

    for episode in tqdm(range(1, settings.EPISODES - load_episode + 1), ascii=True, unit='episodes'):

        # episode += 1
        # Env.collision_hist = [] -> REDUNDANT???
        episode_reward = 0
        explore = 0
        step = 1
        Action = [np.array([0, 0])]
        v_kmh = [0]

        # Reset environment and get initial state
        current_state = Env.reset()
        Agent.resetRandomProcess()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            action = []

            if np.random.random() > settings.epsilon:
                action = Agent.chooseAction(current_state)[0]
            else:
                action = Agent.randomAction()[0]
                explore += 1
            new_state, reward, done, _ = Env.step(
                action, Action[-1], v_kmh[-1])
            Action.append(action)
            v_kmh.append(new_state[-1])
            time.sleep(1/FPS)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            Agent.replayMemory.append(
                current_state, action, reward, new_state, done)

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy Agents
        for actor in Env.actor_list:
            actor.destroy()

        Agent.train()

        if settings.epsilon > settings.MIN_EPSILON:
            settings.epsilon *= settings.EPSILON_DECAY
            settings.epsilon = max(settings.MIN_EPSILON, settings.epsilon)

        print('Episode :{}, Step :{}, Epsilon :{}, Reward :{}, Explore_rate :{}, Avg_v_kmh :{}, Max_v_kmh :{}'
              .format(episode+load_episode, step, settings.epsilon, episode_reward, explore/step, np.array(v_kmh).mean(), np.array(v_kmh).max()))

        ep_rewards.append(episode_reward)
        ep.append(episode+load_episode)
        Step.append(step)
        Explore.append(explore)
        Epsilon.append(settings.epsilon)
        avg = ((avg*(episode+load_episode)+episode_reward) /
               (episode+load_episode+1))
        avg_reward.append(avg)

        if (episode+load_episode) % 100 == 0:
            df = pd.DataFrame({'Episode': ep, 'Reward': ep_rewards, 'avg_reward': avg_reward, 'Step': Step, 'Explore': Explore, 'PCT_Explore': np.array(
                Explore)/np.array(Step)*100, 'Epsilon': Epsilon, 'Avg_velocity': np.array(v_kmh).mean(), 'Max_velocity': np.array(v_kmh).max()})
            # if LOAD == True:
            #     df = pd.concat([df_load, df], ignore_index=True)
            save_model('JAN_27_0.99995', episode+load_episode, df, Agent)

    # Agent.terminate = True
    # trainer_thread.join()

    # close_carla()
