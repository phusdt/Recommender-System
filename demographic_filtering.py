import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from get_data import (
    get_users_data,
    get_rating_base_data,
    get_rating_test_data,
)


class DF(object):
    """ Docstring for DF """

    def __init__(self, users, Y_data, k, dist_func=cosine_similarity):
        self.users = users
        self.Y_data = Y_data
        self.k = k
        self.dist_func = dist_func

        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

        self.Ybar_data = None
    def _get_users_features(self):
        """
        convert demographic data of user to binary
        """
        self.users_features = self.users.copy()
        # First convert sex: if M == 1 else F == 0
        self.users_features["sex"] = self.users_features.sex.map(
            lambda x: 1.0 * (x == "M")
        )
        self.users_features["male"] = np.where(self.users_features["sex"] == 1.0, 1.0, 0.0)
        self.users_features["female"] = np.where(self.users_features["sex"] == 0.0, 1.0, 0.0)
        # Then i convert age follow:
        # 1:  "Under 18"
        # 18:  "18-24"
        # 25:  "25-34"
        # 35:  "35-44"
        # 45:  "45-49"
        # 50:  "50-55"
        # 56:  "56+"
        self.users_features["age"] = self.users_features.age.map(
            lambda x: 1
            if int(x) >= 1 and int(x) < 18
            else (
                18
                if int(x) >= 18 and int(x) < 25
                else (
                    25
                    if int(x) >= 25 and int(x) < 35
                    else (
                        35
                        if int(x) >= 35 and int(x) < 45
                        else (
                            45
                            if int(x) >= 45 and int(x) < 50
                            else (50 if int(x) >= 50 and int(x) < 56 else 56)
                        )
                    )
                )
            )
        )