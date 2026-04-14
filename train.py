from env import StudentEnv
from agent import Agent
import numpy as np

env = StudentEnv(ability=0.8, speed=0.3)
agent = Agent(state_dim=4, action_dim=4)

episodes = 200

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(100):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.store((state, action, reward, next_state))
        agent.train()

        state = next_state
        total_reward += reward

    print(f"Episode {ep}, Reward: {total_reward}")