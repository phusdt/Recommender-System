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