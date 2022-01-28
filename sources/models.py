from keras.models import Sequential
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.models import model_from_json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as backend


def defineActor(state_space_size, action_space_size):

    actor_initializer = tf.random_uniform_initializer(
        minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(state_space_size))   # self.state_space_size

    nextLayer = layers.BatchNormalization()(inputs)  # Normalize this?
    nextLayer = layers.Dense(200)(nextLayer)
    nextLayer = layers.BatchNormalization()(nextLayer)
    nextLayer = layers.Activation("relu")(nextLayer)

    nextLayer = layers.Dense(100)(nextLayer)
    nextLayer = layers.BatchNormalization()(nextLayer)
    nextLayer = layers.Activation("relu")(nextLayer)

    # tanh maps into the interval of [-1, 1]
    outputs = layers.Dense(action_space_size, activation="tanh",
                           kernel_initializer=actor_initializer)(nextLayer)  # self.action_space_size

    # max_control signal is 1.0 for EACH joint of the Hopper robot
    outputs = outputs * 1.0     # self.max_control_signal
    return tf.keras.Model(inputs, outputs)


def defineCritic(state_space_size, action_space_size):

    critic_initializer = tf.random_uniform_initializer(
        minval=-0.003, maxval=0.003)

    state_inputs = layers.Input(
        shape=(state_space_size))   # self.state_space_size

    state_stream = layers.BatchNormalization()(state_inputs)  # Normalize this?
    state_stream = layers.Dense(200)(state_stream)
    state_stream = layers.BatchNormalization()(state_stream)
    state_stream = layers.Activation("relu")(state_stream)

    action_inputs = layers.Input(
        shape=(action_space_size))  # self.action_space_size

    # Merge the two seperate information streams
    merged_stream = layers.Concatenate()([state_stream, action_inputs])
    merged_stream = layers.Dense(100)(merged_stream)
    merged_stream = layers.Activation("relu")(merged_stream)

    outputs = layers.Dense(
        1, kernel_initializer=critic_initializer)(merged_stream)

    return tf.keras.Model([state_inputs, action_inputs], outputs)
