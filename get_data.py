import numpy as np
import pandas as pd
import gc
import torch

def get_users_data(data, file_name):
    """
    Get demographic data of users
    Output: matrix users
    """
    _user_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
    users = pd.read_csv(
        "./{0}/{1}".format(data, file_name), sep="|", names=_user_cols
    )

    return users
def get_items_data(data, file_name):
    """
    Get items data
    Output: dataframe items
    """
    _item_cols = [
        "movie_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "filmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western",
    ]
    items = pd.read_csv(
        "./{0}/{1}".format(data, file_name),
        sep="|",
        names=_item_cols,
        encoding="latin-1",
    )

    return items

def get_items_data():
    """
    Get items data
    Output: dataframe items
    """
    _item_cols = [
        "movie_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "filmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western",
    ]
    items = pd.read_csv('./ml-100k/u.item', sep='|', names=_item_cols, encoding='latin-1')

    return items

def get_rating_test_data():
    """
    Get rating_test data
    Output: data frame rating_test
    """
    _rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    rating_test = pd.read_csv('./ml-100k/u2.test', sep='\t', names=_rating_cols, encoding='latin-1')

    return rating_test

def get_rating_base_data():
    """
    Get rating_base data
    Output: dataframe rating_base
    """
    _rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    rating_base = pd.read_csv('./ml-100k/u2.base', sep='\t', names=_rating_cols, encoding='latin-1')
    return rating_base

def convert(data, num_users, num_movies):
    '''
        Convert data
    '''
    new_data = []

    for id_user in range(1, num_users+1):
        id_movie = data[:, 1][data[:, 0]==id_user]
        id_rating = data[:, 2][data[:,0]==id_user]
        ratings = np.zeros(num_movies, dtype=np.uint32)
        ratings[id_movie - 1] = id_rating
        new_data.append(ratings)
    
        del id_movie
        del id_rating
        del ratings
    # create torch tensor
    new_data = torch.FloatTensor(new_data)
    # mapping to binary rating
    new_data[new_data == 0] = -1
    new_data[new_data == 1] = 0
    new_data[new_data == 2] = 0
    new_data[new_data >= 3] = 1

    return new_data

def get_dataset_100k():
    '''
        Get data for RBM
    '''
    training_set=pd.read_csv('./ml-100k/ua.base', delimiter='\t')
    training_set=np.array(training_set, dtype=np.uint32)

    test_set=pd.read_csv('./ml-100k/ua.test', delimiter='\t')
    test_set=np.array(test_set, dtype=np.uint32)

    num_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
    num_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))

    training_set=convert(training_set,num_users, num_movies)
    test_set=convert(test_set,num_users, num_movies)

    return training_set, test_set

def get_rating_data(data, file_name):
    """
    Get rating data
    Output: dataframe rating
    """
    _rating_cols = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(
        "./{0}/{1}".format(data, file_name),
        sep="\t",
        names=_rating_cols,
        encoding="latin-1",
    )
    ratings.drop(["timestamp"], axis=1, inplace=True)
    return ratings