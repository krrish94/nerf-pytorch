"""
Utils to compute metrics and track them across training.
"""


class ScalarMetric(object):
    def __init__(self):
        self.value = 0.0
        self.num_observations = 0.0
        self.aggregated_value = 0.0
        self.reset()

    def reset(self):
        self.value = []
        self.num_observations = 0.0
        self.aggregated_value = 0.0

    def __repr__(self):
        return str(self.peek())

    def update(self, x):
        self.aggregated_value += x
        self.num_observations += 1

    def peek(self, x):
        return self.aggregated_value / (
            self.num_observations if self.num_observations > 0 else 1
        )
