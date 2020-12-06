import numpy as np
import random
from get_data import(
    get_users_data,
    get_rating_base_data,
    get_rating_test_data,
)

class Perceptron:
    def __init__(self, dataset, learning_rate, n_iters):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w1 = random.uniform(0, 1)
        self.w2 = random.uniform(0, 1)

    def fit(self):

        n = self.dataset[0, 0] * self.w1 + self.dataset[0, 1] * self.w2

        for i in range(1, self.n_iters):
            # update weight
            w1_stamp = self.w1 + self.learning_rate * self.dataset[i - 1, 0] * (self.dataset[i - 1, 2] - n)
            w2_stamp = self.w2 + self.learning_rate * self.dataset[i - 1, 1] * (self.dataset[i - 1, 2] - n)
            if np.abs(self.w1 - w1_stamp) <= 0.0001 and np.abs(self.w2 - w2_stamp) <= 0.0001:
                break
            else:
                self.w1 = w1_stamp
                self.w2 = w2_stamp

            n = self.dataset[i, 0] * self.w1 + self.dataset[i, 1] * self.w2
        
    def predict(self):
        """Predict rating based on w"""
        new_predicted_ratings = []
        for row in self.dataset:
            new_predicted_rating = row[0] * self.w1 + row[1] * self.w2
            new_predicted_ratings.append(new_predicted_rating)
        return new_predicted_ratings
