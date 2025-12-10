import torch
from pathlib import Path
from env_utilities import create_environment
from models.MLP_baseline import MLPBase
import json

"""
To run evaluation, paste the following command into repo root after training:
python -m training_scripts.evaluate_MLP_base
"""

def identify_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_simulation(env, model, device, max_steps=500, render=True):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        state = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(state)
            action = logits.argmax(dim=-1).item()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if render:
            pass
        done = terminated or truncated
    return total_reward, steps


def main():
    device = identify_device()
    print(f"Evaluating on: {device}")

    model = MLPBase(state_in=4, hidden_lay=64, action_out=2).to(device)
    model_path = Path("raw_data/mlp_baseline_cartpole.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded trained model from: {model_path}")

    env = create_environment(render=True)

    num_episodes = 20
    total_reward_sum = 0.0
    total_num_steps = 0
    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        episode_reward, num_steps = run_simulation(
            env, model, device, max_steps=500, render=True)
        print(
            f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {num_steps}")
        total_reward_sum += episode_reward
        total_num_steps += num_steps
        episode_rewards.append(episode_reward)
        episode_steps.append(num_steps)
    env.close()
    avg_reward = total_reward_sum / num_episodes
    avg_steps = total_num_steps / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    print(f"Average Steps over {num_episodes} episodes: {avg_steps}")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    eval_data = {
        "model_name": "snn_single_step_cartpole",
        "num_episodes": num_episodes,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
    }

    out_path = results_dir / "eval_MLP_baseline.json"
    with open(out_path, "w") as f:
        json.dump(eval_data, f, indent=2)

if __name__ == "__main__":
    main()
