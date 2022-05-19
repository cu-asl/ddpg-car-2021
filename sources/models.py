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

import settings


def defineActor(state_space_size, action_space_size):

    actor_initializer = tf.random_uniform_initializer(
        minval=-0.003, maxval=0.003)

    img_inputs = layers.Input(shape=(settings.IM_HEIGHT, settings.IM_WIDTH, 2))

    img_stream = layers.Conv2D(64, (3, 3), padding='same')(img_inputs)
    img_stream = layers.Activation("relu")(img_stream)
    img_stream = layers.AveragePooling2D(pool_size=(
        5, 5), strides=(3, 3), padding='same')(img_stream)
    img_stream = layers.BatchNormalization()(img_stream)

    img_stream = layers.Conv2D(64, (3, 3), padding='same')(img_stream)
    img_stream = layers.Activation("relu")(img_stream)
    img_stream = layers.AveragePooling2D(pool_size=(
        5, 5), strides=(3, 3), padding='same')(img_stream)
    img_stream = layers.BatchNormalization()(img_stream)

    img_stream = layers.Conv2D(64, (3, 3), padding='same')(img_stream)
    img_stream = layers.Activation("relu")(img_stream)
    img_stream = layers.AveragePooling2D(pool_size=(
        5, 5), strides=(3, 3), padding='same')(img_stream)
    img_stream = layers.BatchNormalization()(img_stream)

    img_stream = layers.Flatten()(img_stream)

    img_stream = layers.Dense(4)(img_stream)

    #################################################################################################

    non_img_inputs = layers.Input(
        shape=(state_space_size-1))   # -1 for img input

    #################################################################################################

    merged_stream = layers.Concatenate()([img_stream, non_img_inputs])

    merged_stream = layers.BatchNormalization()(merged_stream)
    merged_stream = layers.Dense(200)(merged_stream)
    merged_stream = layers.BatchNormalization()(merged_stream)
    merged_stream = layers.Activation("relu")(merged_stream)

    merged_stream = layers.BatchNormalization()(merged_stream)
    merged_stream = layers.Dense(100)(merged_stream)
    merged_stream = layers.BatchNormalization()(merged_stream)
    merged_stream = layers.Activation("relu")(merged_stream)

    # tanh maps into the interval of [-1, 1]
    outputs = layers.Dense(action_space_size, activation="tanh",
                           kernel_initializer=actor_initializer)(merged_stream)  # self.action_space_size

    # max_control signal is 1.0 for EACH joint of the Hopper robot
    outputs = outputs * 1.0     # self.max_control_signal
    return tf.keras.Model([img_inputs, non_img_inputs], outputs)


def defineCritic(state_space_size, action_space_size):

    critic_initializer = tf.random_uniform_initializer(
        minval=-0.003, maxval=0.003)

    img_inputs = layers.Input(shape=(settings.IM_HEIGHT, settings.IM_WIDTH, 2))

    img_stream = layers.Conv2D(64, (3, 3), padding='same')(img_inputs)
    img_stream = layers.Activation("relu")(img_stream)
    img_stream = layers.AveragePooling2D(pool_size=(
        5, 5), strides=(3, 3), padding='same')(img_stream)
    img_stream = layers.BatchNormalization()(img_stream)

    img_stream = layers.Conv2D(64, (3, 3), padding='same')(img_stream)
    img_stream = layers.Activation("relu")(img_stream)
    img_stream = layers.AveragePooling2D(pool_size=(
        5, 5), strides=(3, 3), padding='same')(img_stream)
    img_stream = layers.BatchNormalization()(img_stream)

    img_stream = layers.Conv2D(64, (3, 3), padding='same')(img_stream)
    img_stream = layers.Activation("relu")(img_stream)
    img_stream = layers.AveragePooling2D(pool_size=(
        5, 5), strides=(3, 3), padding='same')(img_stream)
    img_stream = layers.BatchNormalization()(img_stream)

    img_stream = layers.Flatten()(img_stream)

    img_stream = layers.Dense(4)(img_stream)

    #################################################################################################

    non_img_inputs = layers.Input(
        shape=(state_space_size-1))   # self.state_space_size

    state_stream = layers.Concatenate()([img_stream, non_img_inputs])

    state_stream = layers.BatchNormalization()(state_stream)  # Normalize this?
    state_stream = layers.Dense(200)(state_stream)
    state_stream = layers.BatchNormalization()(state_stream)
    state_stream = layers.Activation("relu")(state_stream)

    #################################################################################################

    action_inputs = layers.Input(
        shape=(action_space_size))  # self.action_space_size

    #################################################################################################

    merged_stream = layers.Concatenate()([state_stream, action_inputs])

    merged_stream = layers.Dense(100)(merged_stream)
    merged_stream = layers.Activation("relu")(merged_stream)

    outputs = layers.Dense(
        1, kernel_initializer=critic_initializer)(merged_stream)

    return tf.keras.Model([img_inputs, non_img_inputs, action_inputs], outputs)
