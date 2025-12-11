import torch
from pathlib import Path
from env_utilities import create_environment 
from models.SNN_single import SNN_SingleStep
import json
"""
To run evaluation, paste the following command into repo root after training:
python -m result_scripts.evaluate_SNN_single
"""

def identify_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_simulation(env, model, device, max_steps = 500, render = True):
    obs, info = env.reset()
    final_reward = 0.0
    num_steps = 0

    done = False
    while not done and num_steps < max_steps:
        state = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.action(state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        final_reward += reward
        num_steps += 1

        done = terminated or truncated
        if render:
            pass
    return final_reward, num_steps

def main():
    device = identify_device()
    print(f"Evaluating on: {device}")
    model = SNN_SingleStep(state_in=4, hidden_lay=64, action_out=2).to(device)
    model_path = Path("raw_data/snn_single_step_cartpole.pt")
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

    out_path = results_dir / "eval_snn_single_step.json"
    with open(out_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"Saved evaluation metrics to {out_path}")

if __name__ == "__main__":
    main()

