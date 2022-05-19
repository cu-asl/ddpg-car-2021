# CARLA environment settings
SECONDS_PER_EPISODE = 1_000
IM_WIDTH = 240
IM_HEIGHT = 180
SHOW_CAM = True    # Preview the camera - You can manually set it to True.
NUMBER_OF_AUTO_VEHICLES = 0

# Agent settings
NOISE = True
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001

# Trainer settings
EPISODES = 40_000
MINIBATCH_SIZE = 32
UPDATE_TARGET_EVERY = 1
MEMORY_FRACTION = 0.6
SAVE_MODEL_EVERY = 100

# DDPG settings
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000

# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.99995  # 0.9975 0.99975
MIN_EPSILON = 0.001

# Load model
# If you want to load model to continue training from previous training session, change this to True.
LOAD = False
LOADED_ACTOR_NAME = 'actorFEB_3_20000'
LOADED_CRITIC_NAME = 'criticFEB_3_20000'
LOADED_CSV_NAME = 'FEB_3_20000'

# Test settings
TEST_EPISODES = 100
ACTOR_NAME = 'actorModel 5 - MAY_12_27800'
CRITIC_NAME = 'criticModel 5 - MAY_12_27800'
CSV_NAME = 'Model 5 - MAY_12_27800'
