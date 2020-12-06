import numpy as np
import pandas as pd

def get_users_data():
    """
    Get demographic data of users
    Output: matrix users
    """
    _user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv("./ml-100k/u.user", sep="|", names=_user_cols)

    return users

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
    _rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    rating_test = pd.read_csv('./ml-100k/u2.test', sep='\t', names=_rating_cols, encoding='latin-1')
    
    return rating_test 

def get_rating_base_data():
    """
    Get rating_base data
    Output: dataframe rating_base
    """
    _rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    rating_base = pd.read_csv('./ml-100k/u2.base', sep='\t', names=_rating_cols, encoding='latin-1')
    return rating_base