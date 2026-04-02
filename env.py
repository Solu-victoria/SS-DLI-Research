import numpy as np

class StudentEnv:
    def __init__(self, ability, speed):
        self.ability = ability
        self.speed = speed

        self.fatigue = 0.2
        self.engagement = 0.8

    def reset(self):
        self.fatigue = 0.2
        self.engagement = 0.8
        return self._get_state()

    def step(self, action):
        difficulty = action / 3  # scale 0–1

        # Probability of correct answer
        prob_correct = self._sigmoid(
            self.ability + 0.3*self.engagement - 0.4*self.fatigue - difficulty
        )

        correct = np.random.rand() < prob_correct

        # Response time
        response_time = (1.5 / self.speed) + np.random.normal(0, 0.1)

        # Update fatigue
        self.fatigue += 0.05
        if action == 0:  # easy
            self.fatigue -= 0.1

        # Update engagement
        if abs(difficulty - self.ability) < 0.2:
            self.engagement += 0.05
        else:
            self.engagement -= 0.05

        # Clip values
        self.fatigue = np.clip(self.fatigue, 0, 1)
        self.engagement = np.clip(self.engagement, 0, 1)

        # Reward
        reward = 1 if correct else -0.5
        reward -= 0.2 * self.fatigue

        next_state = self._get_state(response_time, correct)

        return next_state, reward, False, {}

    def _get_state(self, response_time=0, correct=1):
        return np.array([
            correct,
            response_time,
            self.fatigue,
            self.engagement
        ])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

