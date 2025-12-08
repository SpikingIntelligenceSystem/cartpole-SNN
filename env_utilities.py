import gymnasium as gym
import numpy as np


def create_environment(render=False):
    # If render is true, will open a window to visualize environment.
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")
        return env


def random_episode(env, max_steps=500):
    # Run a random episode in the environment and return reward.
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    for t in range(max_steps):
        action = env.action_space.sample()  # Sample random action
        obs, reward, terminated, trunicated, info = env.step(action)
        final_reward += reward
        steps += 1
        done = terminated or trunicated
        if done:
            break
    return final_reward, steps
