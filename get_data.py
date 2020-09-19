import numpy as np
import pandas as pd

def get_user_data():
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