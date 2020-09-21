import numpy as np
from scipy import sparse
from scipy.stats import pearsonr 
from get_data import(
    get_rating_base_data,
    get_rating_test_data,
)

import warnings
warnings.filterwarnings('ignore')

class CF(object):
    """Docstring for DF"""
    def __init__(self, Y_data, k, dist_func=pearsonr):
        self.Y_data = Y_data
        self.k = k
        self.dist_func = dist_func

        #number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

        self.Ybar_data = None #normalized
    
    def _normalize_Y(self):
        """
        Normalize data rating of users
        """
        self.Ybar_data = self.Y_data.copy().astype('float64')
        users = self.Y_data[:, 0]  # all users - first col of the Y_data

        self.Ybar_data = self.Y_data.copy()
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
    
    def _calc_similarity(self):
        """
        Calculate sim values of user with all users
        """
        Ybar_copy = self.Ybar.copy().toarray()
        self.S = []
        for u in range(self.n_users):
            sims = []
            for n in range(self.n_users):
                sim = pearsonr(Ybar_copy[u, :], Ybar_copy[n, :])
                if np.isnan(sim[0]):
                    sims.append(0)
                else:
                    sims.append(sim[0])
            self.S.append(sims)
        self.S = np.round(np.asarray(self.S).astype("float"), 2)

    def fit(self):
        """
        Normalize data and calculate similarity matrix again
        (after some few ratings added)
        """
        self._normalize_Y()
        self._calc_similarity()
    
    def pred(self, u, i):
        """
        Predict the rating of user u for item i
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2:
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:]
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def recommend(self, u):
        """
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet
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
        return np.asarray(predicted_ratings).astype('float64')

    def display(self):
        """
        Display all items which should be recommend for each user
        """
        for u in range(self.n_users):
            predicted_ratings = self.recommend(u)
            predicted_ratings = predicted_ratings[predicted_ratings[:, 2]].argsort
            print('Recommendation: {0} for user {1}'.format(predicted_ratings[:, 1], u))

################################################################################################################################            
RATE_TRAIN = get_rating_base_data().values # convert to matrix
RATE_TEST = get_rating_test_data().values # convert to matrix

RATE_TRAIN[:, :2] -= 1 # start from 0
RATE_TEST[:, :2] -= 1

CF = CF(RATE_TRAIN, k=25)
CF.fit()

print('Similar Matrix Works')
print(CF.S)
print('Number of rows: ', CF.S.shape[0])
print('Number of columns: ', CF.S.shape[1])

ids = np.where(RATE_TEST[:, 0] == 0)[0].astype('int32')
real_items_1 = RATE_TEST[(np.where((RATE_TEST[:, 0] == 0) & (RATE_TEST[:, 2] >= 3)))]
predicted_items = []

for row in RATE_TEST[ids, :]:
    predicted_rating = CF.pred(0, row[1])
    if predicted_rating >= 3:
        predicted_items.append(row[1])

print('Items which user 1 actually like: ', real_items_1[:, 1])
print('Items in prediction which user 1 might like ', predicted_items)


n_test = RATE_TEST.shape[0]
correct_items_count = 0
real_items_user_like_count = len(np.where(RATE_TEST[:, 2] >= 3)[0].astype(np.int32))

user_id = 0
while user_id < CF.n_users:
    ids = np.where(RATE_TEST[:, 0] == user_id)[0].astype('int32')
    real_items = RATE_TEST[(np.where((RATE_TEST[:, 0] == user_id) & (RATE_TEST[:, 2] >= 3)))]
    for row in RATE_TEST[ids, :]:
        predicted_rating = CF.pred(user_id, row[1])
        if predicted_rating >= 3 and row[1] in real_items:
            correct_items_count += 1
    user_id += 1

print('The accuracy of Collaborative Filtering: {0}/{1} '.format(correct_items_count, real_items_user_like_count))