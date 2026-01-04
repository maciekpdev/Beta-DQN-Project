import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make("MountainCar-v0")

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1
)

model.learn(total_timesteps=200_000)
model.save("sb3_dqn_mountaincar")
