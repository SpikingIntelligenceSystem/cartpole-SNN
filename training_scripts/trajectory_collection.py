import torch
from pathlib import Path
from env_utilities import create_environment

"""
To run data collection, paste:
py -m training_scripts.trajectory_collection
Into repo root.
"""


def heuristic(obs):
    x, x_dot, theta, theta_dot = obs
    return 1 if theta > 0 else 0


def collect_trajectories(num_episodes=50, max_steps=500, render=False):
    env = create_environment(render=render)
    states = []
    actions = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        for t in range(max_steps):
            action = heuristic(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            states.append(obs)
            actions.append(action)
            obs = next_obs
            if terminated or truncated:
                break
        print(f"Episode {episode +1}/{num_episodes} complete.")
    env.close()

    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    print(f"Collected {states_tensor.shape[0]} transitions.")
    return states_tensor, actions_tensor


def main():
    data_dir = Path("raw_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / "cartpole_trajectories.pt"

    states, actions = collect_trajectories(
        num_episodes=500, max_steps=500, render=False)
    torch.save(
        {
            "states": states,
            "actions": actions,
        },
        save_path,
    )
    print(f"Data saved to {save_path.resolve()}")


if __name__ == "__main__":
    main()
