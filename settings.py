# CARLA environment settings
SECONDS_PER_EPISODE = 100

# Agent settings
NOISE = False
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001

# Trainer settings
EPISODES = 50_000  # 100
MINIBATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 1
MODEL_NAME = "64x3_CNN"  # "64x3_CNN", "Xception"
MIN_REWARD = -1.5
MEMORY_FRACTION = 0.6
AGGREGATE_STATS_EVERY = 10

# DDPG settings
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000

# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.99995  # 0.9975 99975
MIN_EPSILON = 0.001

# Console settings
SHOW_PREVIEW = False  # Preview the camera - You can manually set it to TRUE.