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
    
    