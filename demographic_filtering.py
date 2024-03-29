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
        # self.users_features['male'] = self.users_features['sex'].map({'M': "1", 'F': "0"})
        self.users_features.drop(["zip_code", "sex"], axis=1, inplace=True)  # we dont need it

        # The get_dummies() function is used to convert categorical variable
        # into dummy/indicator variables.
        self.users_features = pd.get_dummies(
            self.users_features, columns=["age", "occupation"]
        )
        # i set index of users_features dataframe is user_id
        self.users_features.set_index("user_id", inplace=True)
        self.u = self.users_features

    def _calc_similarity(self):
        """
        calculate sim values of user with all users
        """
        # now i convert from dataframe to array for calculate cosine

        self.users_features = self.users_features.to_numpy()
        #self.users_features = self.users_features * 5
        # calculate similarity
        self.similarities = self.dist_func(self.users_features, self.users_features)

    def _normalize_Y(self):
        """
        normalize data rating of users
        """
        self.Ybar_data = self.Y_data.copy()
        self.Ybar_data = self.Ybar_data.astype("float64")

        users = self.Y_data[:, 0]
        self.mu = np.zeros((self.n_users,))

        for n in range(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)

            # and the corresponding ratings
            ratings = self.Y_data[ids, 2]

            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m

            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix(
            (self.Ybar_data[:, 2], (self.Ybar_data[:, 1], self.Ybar_data[:, 0])),
            (self.n_items, self.n_users),
        )
        self.Ybar = self.Ybar.tocsr()

    def fit(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self._get_users_features()
        self._calc_similarity()
        self._normalize_Y()

    def pred(self, u, i):
        """
        predict the rating tof user u for item i
        """
        # find users rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)

        # find similarity btw current user and others
        # who rated i
        sim = self.similarities[u, users_rated_i]

        # find the k most similarity users
        a = np.argsort(sim)[-self.k:]

        nearest_s = sim[a]

        # ratings of nearest users rated item i
        r = self.Ybar[i, users_rated_i[a]]

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def recommend(self, u):
        """
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        predicted_ratings = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted = self.pred(u, i)
                if predicted > 0:
                    new_row = [u, i, predicted]
                    predicted_ratings.append(new_row)
        return np.asarray(predicted_ratings).astype("float64")

    def display(self):
        """
        Display all items which should be recommend for each user
        """
        for u in range(self.n_users):
            predicted_ratings = self.recommend(u)
            predicted_ratings = predicted_ratings[predicted_ratings[:, 2].argsort(kind='quicksort')[::-1]]
            print("Recommendation: {0} for user {1}".format(predicted_ratings[:, 1], u))
    
    def recommend_all_users(self, data):
        """
        Return matrix with predict and real rating for all user
        """
        result = np.empty((0, 3))

        for user in list(set(data[:,0])):
            # get 2d array rating of current user [u, i, rating]
            ids = np.where(data[:, 0] == user)[0]
            items_rated_by_u = data[ids]

            # create empty 2d array predict rating of user
            predict_ratings_u = np.empty((0, 3))

            items_not_rate = [
                x
                for x in range(self.n_items)
                if x not in items_rated_by_u[:, 1]
            ]

            for item in items_not_rate[:]:
                predict_rating = self.pred(user, item)
                # append new row predict rating data into array
                predict_ratings_u = np.append(
                    predict_ratings_u, [[user, item, predict_rating]], axis=0
                )

            # now we have real and predict rating of current user
            # i will sort predict rating data and get top 100
            # result : top 100 predict rating + real rating (from rate test data)

            predict_ratings_u_sorted = predict_ratings_u[
                predict_ratings_u[:, 2].argsort(kind="quicksort")[::-1][0:100]
            ]
            result = np.append(result, items_rated_by_u, axis=0)
            result = np.append(result, predict_ratings_u_sorted, axis=0)

        return result
