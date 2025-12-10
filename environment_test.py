import time
from env_utilities import create_environment, random_episode

"""
Optional test sctipt to verify environment functionality. Run using:
python -m training_scripts.environment_test
In repo root.
"""


def main():
    render = True

    env = create_environment(render=render)

    num_episodes = 5
    for episode in range(1, num_episodes + 1):
        final_reward, steps = random_episode(env)
        print(
            f"Episode {episode}/// Total Reward: {final_reward:.2f} in {steps} steps.")

        if render:
            time.sleep(0.75)  # Pause between episodes
    env.close()


if __name__ == "__main__":
    main()
