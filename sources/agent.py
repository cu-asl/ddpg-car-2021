import tensorflow as tf
import settings
import numpy as np
import time

from replayBuffer import replayBuffer
from sources import models


class DDPGAgent:
    def __init__(self, loaded_actor=None, loaded_critic=None):
        self.discount = settings.DISCOUNT
        self.memorySize = settings.REPLAY_MEMORY_SIZE
        self.batch_size = settings.MINIBATCH_SIZE   # Minibatch size for training

        self.max_control_signal = 1.0
        self.lowerLimit = -1.0
        self.upperLimit = 1.0

        self.polyak_rate = 0.001
        self.action_space_size = 2  # Size of the action vector
        self.state_space_size = 10  # Size of the observation vector

        self.replayMemory = replayBuffer(
            self.memorySize, self.state_space_size, self.action_space_size)

        self.useNoise = settings.NOISE

        if loaded_actor == None and loaded_critic == None:
            self.actor = self.defineActor()
            self.actor_target = self.defineActor()

            self.critic = self.defineCritic()
            self.critic_target = self.defineCritic()

        elif loaded_actor != None and loaded_critic != None:
            self.actor = loaded_actor
            self.actor_target = loaded_actor

            self.critic = loaded_critic
            self.critic_target = loaded_critic

        else:
            print(
                'You need to load both actor and critic, or else, do not load anything.')

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # Actor's learning rate should be smaller
        self.critic_learning_rate = settings.CRITIC_LEARNING_RATE
        self.actor_learning_rate = settings.ACTOR_LEARNING_RATE

        self.critic_optim = tf.keras.optimizers.Adam(self.critic_learning_rate)
        self.actor_optim = tf.keras.optimizers.Adam(self.actor_learning_rate)

        # Random Process Hyper parameters
        std_dev = 0.2
        self.init_noise_process(average=np.zeros(self.action_space_size), std_dev=float(
            std_dev) * np.ones(self.action_space_size))

        self.target_update_counter = 0
        self.training_initialized = False
        self.terminate = False

    def defineActor(self):
        return models.defineActor(self.state_space_size, self.action_space_size)

    def defineCritic(self):
        return models.defineCritic(self.state_space_size, self.action_space_size)

    def update_target(self, target_weights, weights, polyak_rate):
        for (target, weight) in zip(target_weights, weights):
            target.assign(weight * polyak_rate + target * (1 - polyak_rate))

    """Initialize an Ornstein–Uhlenbeck Process to generate correlated noise"""

    def init_noise_process(self, average, std_dev, theta=0.15, dt=0.01, x_start=None):

        self.theta = theta
        self.average = average
        self.std_dev = std_dev
        self.dt = dt
        self.x_start = x_start
        self.x_prior = np.zeros_like(self.average)

    """Generate another instance of the Ornstein–Uhlenbeck process"""

    def noise(self):
        noise = (self.x_prior + self.theta * (self.average - self.x_prior) * self.dt +
                 self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.average.shape))

        self.x_prior = noise
        return noise

    """End Ornstein–Uhlenbeck process and start a new one"""

    def resetRandomProcess(self):
        std_dev = 0.2
        self.init_noise_process(average=np.zeros(self.action_space_size), std_dev=float(
            std_dev) * np.ones(self.action_space_size))

    def chooseAction(self, state):

        img_state = tf.expand_dims(tf.convert_to_tensor(state[0]), 0)
        non_img_state = tf.expand_dims(
            tf.convert_to_tensor(state[1:].astype(np.float64)), 0)
        action = self.actor([img_state, non_img_state])

        noise = self.noise()
        if (self.useNoise == True):
            action = action.numpy() + noise
        else:
            action = action.numpy()

        # Make sure action is withing legal range
        action = np.clip(action, self.lowerLimit, self.upperLimit)

        return [np.squeeze(action)]

    def randomAction(self):
        # Assume the action space is symmetric, for each entry in action vector
        return (np.random.rand(1, self.action_space_size) - self.upperLimit/2.0) * self.upperLimit

    @tf.function
    def update(self, img_states, states, actions, rewards, next_img_states, next_states, isTerminals):

        with tf.GradientTape() as tape:
            target_actions = self.actor_target(
                [next_img_states, next_states], training=True)
            newVector = tf.cast(1 - isTerminals, dtype=tf.float32)
            predicted_values = rewards + self.discount * newVector * \
                self.critic_target(
                    [next_img_states, next_states, target_actions], training=True)

            critic_value = self.critic(
                [img_states, states, actions], training=True)
            critic_loss = tf.math.reduce_mean(
                tf.math.square(predicted_values - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic_optim.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:

            actions2 = self.actor([img_states, states], training=True)
            critic_value = self.critic(
                [img_states, states, actions2], training=True)
            # Remember to negate the loss!
            actor_loss = tf.math.reduce_mean(-1 * critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optim.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

        # Update the target networks
        self.target_update_counter += 1

        if self.target_update_counter > settings.UPDATE_TARGET_EVERY:
            self.update_target(self.actor_target.variables,
                               self.actor.variables, self.polyak_rate)
            self.update_target(self.critic_target.variables,
                               self.critic.variables, self.polyak_rate)
            self.target_update_counter = 0

    def train(self):

        if self.replayMemory.size < settings.MIN_REPLAY_MEMORY_SIZE:
            return

        img_states, states, actions, rewards, next_img_states, next_states, isTerminals = self.replayMemory.sample(
            self.batch_size)

        # Convert to Tensorflow data types
        img_states = tf.convert_to_tensor(img_states)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.cast(tf.convert_to_tensor(rewards), dtype=tf.float32)
        next_img_states = tf.convert_to_tensor(next_img_states)
        next_states = tf.convert_to_tensor(next_states)
        isTerminals = tf.convert_to_tensor(isTerminals)

        self.update(img_states, states, actions, rewards,
                    next_img_states, next_states, isTerminals)

    def save_critic(self, name):
        model_json = self.critic.to_json()
        with open("{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.critic.save_weights("{}.h5".format(name))
        print("Saved critic to disk")

    def save_actor(self, name):
        model_json = self.actor.to_json()
        with open("{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.actor.save_weights("{}.h5".format(name))
        print("Saved actor to disk")
