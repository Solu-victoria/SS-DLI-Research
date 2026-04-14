from env import StudentEnv
from agent import Agent
import numpy as np

def train_model(multi_timescale=False, episodes=200):
    env = StudentEnv(multi_timescale=multi_timescale)

    state_dim = 6 if multi_timescale else 4
    agent = Agent(state_dim=state_dim, action_dim=4)

    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(100):
            action = agent.select_action(state)
            next_state, reward, _, _ = env.step(action)

            agent.store((state, action, reward, next_state))
            agent.train()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        print(f"{'Multi' if multi_timescale else 'Baseline'} | Episode {ep} | Reward: {total_reward:.2f}")

    return rewards_history


if __name__ == "__main__":
    print("Training Baseline Model...")
    baseline_rewards = train_model(multi_timescale=False)

    print("\nTraining Multi-Timescale Model...")
    multi_rewards = train_model(multi_timescale=True)