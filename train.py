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
from keras.models import model_from_json

from tqdm import tqdm

import settings
from sources import agent
from sources import carla


def save_model(name, n, df, agent):
    file_path = "DATA\\"
    df.to_csv(file_path+'{}_{}.csv'.format(name, n))
    agent.save_critic(file_path+'critic'+name+'_'+str(n))
    agent.save_actor(file_path+'actor'+name+'_'+str(n))


def load_model(actor_name, critic_name, model_name):

    json_file = open('DATA\\{}.json'.format(actor_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_actor = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_actor.load_weights("DATA\\{}.h5".format(actor_name))

    print("Loaded actor from disk")
    loaded_actor.summary()

    json_file = open('DATA\\{}.json'.format(critic_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_critic = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_critic.load_weights("DATA\\{}.h5".format(critic_name))

    print("Loaded critic from disk")
    loaded_critic.summary()

    file_path = "DATA\\"
    df = pd.read_csv(file_path+'{}.csv'.format(model_name))
    episode = df.Episode.tail(1).values[0]
    epsilon = df.Epsilon.tail(1).values[0]
    print('Episode : {} , Epsilon : {} '.format(episode, epsilon))

    return df, episode, epsilon, loaded_actor, loaded_critic


if __name__ == '__main__':

    FPS = 60
    ep_rewards = []
    ep = []
    avg = 0
    avg_reward = []
    Step = []
    Explore = []
    Steer = []
    Epsilon = []
    Map = []
    avg_v_kmh = []
    max_v_kmh = []
    results = []

    random.seed(1)
    np.random.seed(1)

    # Create models folder
    if not os.path.isdir('DATA'):
        os.makedirs('DATA')

    # MODEL LOADER - Skip this if you want to train a new model
    if settings.LOAD == True:
        # In case Train from loaded_model
        actor_name = settings.LOADED_ACTOR_NAME
        critic_name = settings.LOADED_CRITIC_NAME
        model_name = settings.LOADED_CSV_NAME

        df_load, load_episode, loaded_epsilon, loaded_actor, loaded_critic = load_model(
            actor_name, critic_name, model_name)
        Agent = agent.DDPGAgent(loaded_actor, loaded_critic)
        settings.epsilon = loaded_epsilon
        nn = 0
        for i in df_load.Reward:
            avg = ((avg*(nn)+i)/(nn+1))
            # pp.update(float(avg))
            nn += 1
        avg = sum(df_load.Reward)/df_load.shape[0]

    else:
        # Create Agent and environment
        Agent = agent.DDPGAgent()
        load_episode = 0
    ###############

    # Iterate over episodes
    Env = carla.CarEnv()
    Env.test = False

    for episode in tqdm(range(1, settings.EPISODES - load_episode + 1), ascii=True, unit='episodes'):

        episode_reward = 0
        explore = 0
        step = 0
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

            if np.random.random() > settings.epsilon:
                action = Agent.chooseAction(current_state)[0]
            else:
                action = Agent.randomAction()[0]
                explore += 1
            new_state, reward, done, is_finished, _ = Env.step(
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

        # Train model
        Agent.train()

        # Exploration limit
        if settings.epsilon > settings.MIN_EPSILON:
            settings.epsilon *= settings.EPSILON_DECAY
            settings.epsilon = max(settings.MIN_EPSILON, settings.epsilon)

        print('Episode :{}, Step :{}, Epsilon :{}, Reward :{}, Explore_rate :{}, Avg_v_kmh :{}, Max_v_kmh :{}, Map :{}'
              .format(episode+load_episode, step, settings.epsilon, episode_reward, explore/step, np.array(v_kmh).mean(), np.array(v_kmh).max(), Env.map_name))

        # Saving data
        ep_rewards.append(episode_reward)
        ep.append(episode+load_episode)
        Step.append(step)
        Explore.append(explore)
        Epsilon.append(settings.epsilon)
        avg = ((avg*(episode+load_episode)+episode_reward) /
               (episode+load_episode+1))
        avg_reward.append(avg)
        avg_v_kmh.append(np.array(v_kmh).mean())
        max_v_kmh.append(np.array(v_kmh).max())
        Map.append(Env.map_name)

        if is_finished == True:
            results.append('Success')
        else:
            results.append('Failure')

        if (episode+load_episode) % settings.SAVE_MODEL_EVERY == 0:
            df = pd.DataFrame({'Episode': ep, 'Reward': ep_rewards, 'avg_reward': avg_reward, 'Step': Step, 'Explore': Explore, 'PCT_Explore': np.array(
                Explore)/np.array(Step)*100, 'Epsilon': Epsilon, 'Avg_velocity': avg_v_kmh, 'Max_velocity': max_v_kmh, 'Map': Map, 'Results': results})
            save_model('APR_23', episode+load_episode, df, Agent)

    print('Training finished.')
