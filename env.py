import numpy as np

class StudentEnv:
    def __init__(self, multi_timescale=False):
        self.multi_timescale = multi_timescale

        # TRUE (hidden) student properties
        self.true_ability = np.random.uniform(0.3, 0.9)
        self.true_speed = np.random.uniform(0.3, 0.9)

        # dynamic states
        self.fatigue = 0.2
        self.engagement = 0.8

        # long-term estimates (only used if multi-timescale)
        self.ability_estimate = 0.5
        self.speed_estimate = 0.5

        self.prev_correct = 1
        self.prev_response_time = 1.0

    def reset(self):
        self.fatigue = 0.2
        self.engagement = 0.8
        self.prev_correct = 1
        self.prev_response_time = 1.0

        self.ability_estimate = 0.5
        self.speed_estimate = 0.5

        return self._get_state()

    def step(self, action):
        # actions: 0 easy, 1 medium, 2 hard, 3 hint
        difficulty_map = [0.3, 0.5, 0.7, 0.4]
        difficulty = difficulty_map[action]

        # probability of correctness (IRT-inspired)
        prob_correct = self._sigmoid(
            self.true_ability + 0.3*self.engagement - 0.4*self.fatigue - difficulty
        )

        correct = np.random.rand() < prob_correct

        # response time
        response_time = (1.5 / self.true_speed) + np.random.normal(0, 0.1)
        response_time = np.clip(response_time, 0.5, 3.0)

        # update fatigue
        self.fatigue += 0.05
        if action == 0:  # easy reduces fatigue
            self.fatigue -= 0.08

        # update engagement
        if abs(difficulty - self.true_ability) < 0.2:
            self.engagement += 0.05
        else:
            self.engagement -= 0.05

        self.fatigue = np.clip(self.fatigue, 0, 1)
        self.engagement = np.clip(self.engagement, 0, 1)

        # reward (IMPORTANT: same for both models)
        reward = 1 if correct else -0.5
        reward += 0.3 * self.engagement
        reward -= 0.2 * self.fatigue

        # update long-term estimates (ONLY affects state, not reward)
        self.ability_estimate = 0.9 * self.ability_estimate + 0.1 * correct
        self.speed_estimate = 0.9 * self.speed_estimate + 0.1 * (1 / response_time)

        self.prev_correct = int(correct)
        self.prev_response_time = response_time

        return self._get_state(), reward, False, {}

    def _get_state(self):
        if self.multi_timescale:
            return np.array([
                self.prev_correct,
                self.prev_response_time,
                self.fatigue,
                self.engagement,
                self.ability_estimate,
                self.speed_estimate
            ], dtype=np.float32)
        else:
            return np.array([
                self.prev_correct,
                self.prev_response_time,
                self.fatigue,
                self.engagement
            ], dtype=np.float32)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))