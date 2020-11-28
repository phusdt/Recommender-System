import numpy as np
import random
from collaborative_filtering import *
from demographic_filtering import *
from get_data import (
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