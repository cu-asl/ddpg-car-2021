import os
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from keras.models import model_from_json

import settings
from sources import carla
from sources import agent


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

    actor_name = settings.ACTOR_NAME
    critic_name = settings.CRITIC_NAME
    model_name = settings.CSV_NAME
    _, _, _, loaded_actor, loaded_critic = load_model(
        actor_name, critic_name, model_name)

    Agent = agent.DDPGAgent(loaded_actor, loaded_critic)
    Env = carla.CarEnv()

    Env.test = True

    test_ep = settings.TEST_EPISODES
    test_step = []
    test_reward = []
    dist_average = []
    avg_v_kmh = []
    max_v_kmh = []
    Map = []
    results = []

    # Create test results folder
    if not os.path.isdir('Test results'):
        os.makedirs('Test results')

    for episode in tqdm(range(1, test_ep + 1), ascii=True, unit='episodes'):
        episode_reward = 0
        step = 0
        Action = [np.array([0, 0])]
        v_kmh = [0]
        dist_ = []

        # Reset environment and get initial state
        current_state = Env.reset()
        Agent.resetRandomProcess()

        done = False

        while True:

            # Predict an action based on current observation space
            action = Agent.chooseAction(current_state)[0]

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, is_finished, _ = Env.step(
                action, Action[-1], v_kmh[-1])

            # Save testing data
            Action.append(action)
            dist_.append(Env.get_distance())
            v_kmh.append(new_state[-1])

            # Set current step for next loop iteration
            current_state = new_state

            # map_.append(Env.get_location())
            episode_reward += reward
            step += 1

            # If done - agent crashed, break an episode
            if done:
                break

        for actor in Env.actor_list:
            actor.destroy()

        dist_average.append(np.array(dist_).mean())
        avg_v_kmh.append(np.array(v_kmh).mean())
        max_v_kmh.append(np.array(v_kmh).max())
        # map_save.append(map_)
        test_reward.append(episode_reward)
        test_step.append(step)
        Map.append(Env.map_name)

        if is_finished == True:
            results.append('Success')
        else:
            results.append('Failed')

    df_test = pd.DataFrame({'Episode': [i for i in range(1, test_ep+1)],
                            'Step': test_step, 'Reward': test_reward,
                            'Avg_Dist': dist_average, 'Avg_velocity': avg_v_kmh, 'Max_velocity': max_v_kmh, 'Map': Map, 'Results': results})

    name = 'Model_{}_{}eps'.format(time.time(), test_ep)
    file_path = "Test results\\"
    df_test.to_csv(file_path+'{}.csv'.format(name))

    print('Finished testing')
