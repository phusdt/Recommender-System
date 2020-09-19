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
                sim = pearsonr(Y_bar_copy[u, :], Ybar_copy[n, :])
                if np.isnan(sim[0]):
                    sims.append(0)
                else:
                    sims.append(sim[0])
            self.S.append(sims)
        self.S = np.round(np.asarray(self.S).astype('float'), 2)
        

