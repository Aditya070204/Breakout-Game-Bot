import os
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

gym.register_envs(ale_py)

model_path = "./models/pacman_60.zip"
env = gym.make('ALE/Breakout-v5',render_mode='human')
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env.envs[0].unwrapped._game_difficulty = 1
env.envs[0].unwrapped._game_mode = 20
model = PPO.load(model_path)
mean_reward, std_reward = evaluate_policy(env=env, model=model, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
env.close()
