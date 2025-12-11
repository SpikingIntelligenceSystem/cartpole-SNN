import json
from pathlib import Path
import matplotlib.pyplot as plt

"""
To run plotting script, paste the following command into root directory:
python -m result_scripts.simulation_results
"""

def load_evaluation(path: Path):
    with open(path, "r") as f:
        return json.load(f)
    
def compute_data(eval_data):
    rewards = eval_data["episode_rewards"]
    steps = eval_data["episode_steps"]
    num_episodes = len(rewards)
    reward_average = eval_data["avg_reward"]
    steps_average = eval_data["avg_steps"]
    max_steps = max(steps)
    failures = sum(1 for s in steps if s< max_steps)
    failure_rate = failures / num_episodes
    return{
        "avg_reward": reward_average,
        "avg_steps": steps_average,
        "failure_rate": failure_rate,
        "episodes": num_episodes,
        "max_steps": max_steps,
    }
    
def main():
    results_dir = Path("results")
    model_files = {
        "MLP Baseline": results_dir / "eval_MLP_baseline.json",
        "SNN Single": results_dir / "eval_snn_single_step.json",
        "SNN Temporal": results_dir / "eval_snn_temporal.json",
    }

    model_names = []
    avg_rewards = []
    failure_rates = []
    avg_steps = []

    print(f"Loaded evaluation metrics.")
    
    for display_name, path in model_files.items():
        if not path.exists():
            print(f"[SKIP]-{display_name}, file not found in {path}.")
            continue
        eval_data = load_evaluation(path)
        metrics = compute_data(eval_data)

        model_names.append(display_name)
        avg_rewards.append(metrics["avg_reward"])
        failure_rates.append(metrics["failure_rate"])
        avg_steps.append(metrics["avg_steps"])
        
        print(f"Display Name: {display_name}"
              f"Average Reward: {metrics['avg_reward']:.2f}"
              f"Average Steps: {metrics['avg_steps']:.2f}"
              f"Fail Rate: {metrics['failure_rate']*100:.2f}%"
              )
    if not model_names:
        print("No evaluation files found. Check paths or run evaluation scripts before retry.")
        return
    # Begin plot 1 - Reward Average
    plt.figure(figsize=(10,5))
    x=range(len(model_names))
    plt.bar(x,avg_rewards)
    plt.xticks(x, model_names)
    plt.ylabel("Average Reward")
    plt.title("Average Reward Based On Model")

    for i, v in enumerate(avg_rewards):
        plt.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    reward_path = results_dir / "cartpole_avg_rewards_graph.png"
    plt.tight_layout()
    plt.savefig(reward_path, dpi=150)
    plt.close()
    # End plot 1 - Reward Average

    # Begin plot 2 - Failure Rate
    plt.figure(figsize=(10,5))
    failure_pct = [r * 100.0 for r in failure_rates]
    plt.bar(x, failure_pct)
    plt.xticks(x, model_names)
    plt.ylim(0,5)
    plt.ylabel("Failure Percentage")
    plt.title("Failure Rate Based On Model")

    for i, v in enumerate(failure_pct):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    failure_path = results_dir / "cartpole_fail_rate_graph.png"
    plt.tight_layout()
    plt.savefig(failure_path, dpi=150)
    plt.close()
    # End plot 2 - Failure Rate

if __name__ == "__main__":
    main()
