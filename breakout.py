import os
import random
import gymnasium as gym
import ale_py
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import optuna
import torch
from torch.optim.lr_scheduler import LinearLR

gym.register_envs(ale_py)

log_path = "./break_logs"
os.makedirs(log_path, exist_ok=True)
save_path = "./models"
os.makedirs(save_path, exist_ok=True)

# "ALE/Pacman-v5": {'modes': [0, 1, 2, 3, 4, 5, 6, 7], 'difficulties': [0, 1],
                    #   'mean_reward': 3000}

env = gym.make('ALE/Breakout-v5', max_episode_steps=1000, mode = 0, difficulty=0, render_mode=None)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

modes = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
difficulties = [0, 1]

class dynamic_lr:
    def __init__(self, total_timesteps, initial_lr, final_lr):
        self.total_timesteps = total_timesteps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
    
    def update(self, model, num_steps):
        progress = 1 - num_steps/self.total_timesteps
        lr = self.final_lr + progress*(self.initial_lr-self.final_lr)
        for param in model.policy.optimizer.param_groups:
            param['lr'] = lr
        return lr

total_training_steps = 100 * 100000  # Total timesteps across all iterations
initial_lr = 0.0005
final_lr = 0.0001
lr_callback = dynamic_lr(total_training_steps,initial_lr,final_lr)

def optimize_hyperparameters(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_int('n_steps', 512, 2048, step=512)
    gamma = trial.suggest_float("gamma",0.9,0.999)

    model = A2C('CnnPolicy', env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, verbose=0)
    model.learn(total_timesteps=50000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    return mean_reward

# study = optuna.create_study(direction='maximize')
# study.optimize(optimize_hyperparameters, n_trials=20)
# best_params = study.best_trial.params

# initial_lr = best_params['learning_rate']

# model = PPO('CnnPolicy',
#              env, 
#              learning_rate=initial_lr, 
#              verbose=1, 
#              gamma=0.99,  
#              n_steps=2048,
#              tensorboard_log=log_path)
old_model = PPO.load('C:/Notes/ML Projects/Atari/models/pacman_60.zip')
model = PPO('CnnPolicy', env, gamma=0.99, clip_range=0.2, learning_rate=0.0005, verbose=1, tensorboard_log=log_path, device='cuda')
model.policy.load_state_dict(old_model.policy.state_dict())
# optimizer = model.policy.optimizer
# scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=70)

# unwrapped_env = env.envs[0].unwrapped
# print(unwrapped_env)
# print("Attributes of the environment:")
# for attr in dir(unwrapped_env):
#     print(attr)

j, k = 0, 0
for i in range(1,101):
    model.learn(total_timesteps=100000)
      # This updates the learning rate after each iteration
    new_lr = lr_callback.update(model,i*100000)
    
    # Evaluate policy performance
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Iteration {i}: Mean Reward = {mean_reward}, Std Reward = {std_reward}")
    print(f"Updated learning rate: {new_lr}")

    with open("training_rewards.txt", "a") as f:
        f.write(f"Iteration {i}: Mean Reward = {mean_reward}, Std Reward = {std_reward}\n")

    # if mean_reward > 200 and j==len(modes)-1 and k==len(difficulties)-1:
    #     print("Achieved target performance and reached final mode/difficulty. Stopping training.")
    #     break

    if i%20==0:
        difficulty = random.choice(difficulties)
        env.envs[0].unwrapped._game_difficulty = difficulty
        print(f"New Difficulty: {difficulty}")

    if i%20==0:
        model.save(f"{save_path}/breakout_{i}")
model.save(f"{save_path}/breakout")