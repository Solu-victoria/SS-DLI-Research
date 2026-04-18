import numpy as np

#tired but exploring now
class StudentEnv:
    def __init__(self, multi_timescale=False):
        self.multi_timescale = multi_timescale

        # TRUE (hidden) student properties
        self.true_ability = np.random.uniform(0.3, 0.9)
        self.true_speed = np.random.uniform(0.3, 0.9)

        # long-term estimates (only used if multi-timescale)
        self.ability_estimate = 0.5
        self.speed_estimate = 0.5

        self.prev_correct = 1
        self.prev_response_time = 1.0

    def reset(self):
        # New student each episode
        self.true_ability = np.random.uniform(0.3, 0.9)
        self.true_speed = np.random.uniform(0.3, 0.9)

        self.prev_correct = 1
        self.prev_response_time = 1.0

        self.ability_estimate = 0.5
        self.speed_estimate = 0.5

        return self._get_state()

    def step(self, action):
        # actions: 0 easy, 1 medium, 2 hard, 3 hint
        difficulty_map = [0.3, 0.5, 0.7, 0.4]
        difficulty = difficulty_map[action]

        # ----------- STOCHASTIC CORRECTNESS -----------

        prob_correct = self._sigmoid(self.true_ability - difficulty)

        # 🔥 add noise so single-step correctness is unreliable
        prob_correct += np.random.normal(0, 0.1)
        prob_correct = np.clip(prob_correct, 0, 1)

        correct = np.random.rand() < prob_correct

        # ----------- RESPONSE TIME (AMBIGUITY) -----------

        # base from speed
        response_time = (1.5 / self.true_speed)

        # 🔥 high ability students may still be slow (thinking deeply)
        if self.true_ability > 0.7:
            response_time += np.random.uniform(0.5, 1.0)

        # noise
        response_time += np.random.normal(0, 0.1)
        response_time = np.clip(response_time, 0.5, 3.0)

        # ----------- REWARD -----------

        reward = 1 if correct else -0.5

        # small penalty for slow responses
        reward -= 0.1 * response_time

        # ----------- LONG-TERM ESTIMATION -----------

        # ability estimate (temporal averaging)
        self.ability_estimate += 0.1 * (correct - self.ability_estimate)

        # speed estimate (inverse response time)
        self.speed_estimate += 0.1 * ((1 / response_time) - self.speed_estimate)

        # ----------- UPDATE OBSERVATIONS -----------

        self.prev_correct = int(correct)
        self.prev_response_time = response_time

        return self._get_state(), reward, False, {}

    def _get_state(self):
        if self.multi_timescale:
            return np.array([
                self.prev_correct,
                self.prev_response_time,
                self.ability_estimate,
                self.speed_estimate
            ], dtype=np.float32)
        else:
            return np.array([
                self.prev_correct,
                self.prev_response_time
            ], dtype=np.float32)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))