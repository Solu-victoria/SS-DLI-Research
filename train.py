from env import StudentEnv
from agent import Agent
import numpy as np

def train_model(multi_timescale=False, episodes=200, ep_steps=100, save_path=None):
    env = StudentEnv(multi_timescale=multi_timescale)

    state_dim = 4 if multi_timescale else 2
    agent = Agent(state_dim=state_dim, action_dim=4)

    rewards_history = []

    for ep in range(episodes+1):
        state = env.reset()
        total_reward = 0

        for t in range(ep_steps):
            action = agent.select_action(state)
            next_state, reward, _, _ = env.step(action)

            agent.store((state, action, reward, next_state))
            agent.train()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        print(f"{'Multi' if multi_timescale else 'Baseline'} | Episode {ep} | Reward: {total_reward:.2f}")

    # Save model
    if save_path:
        agent.save(save_path)

    return rewards_history

def run_train_experiment(num_runs=5, episodes=200, num_steps_per_ep=100):
    all_baseline = []
    all_multi = []

    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")

        # Create filenames
        baseline_path = f"models/v2/{num_steps_per_ep}_steps_per_ep/baseline/run_{run+1}.pth"
        multi_path = f"models/v2/{num_steps_per_ep}_steps_per_ep/multi/run_{run+1}.pth"

        # Train + save
        baseline = train_model(
            multi_timescale=False,
            episodes=episodes,
            ep_steps = num_steps_per_ep,
            save_path=baseline_path,
        )

        multi = train_model(
            multi_timescale=True,
            episodes=episodes,
            ep_steps = num_steps_per_ep,
            save_path=multi_path,
        )

        all_baseline.append(baseline)
        all_multi.append(multi)

    return np.array(all_baseline), np.array(all_multi)