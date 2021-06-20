import numpy as np
import pandas as pd
import plotly.graph_objects as go
import copy
from tmdbv3api import TMDb
from tmdbv3api import Movie
from scipy.spatial.distance import hamming, euclidean, chebyshev, cityblock
import urllib.request


def main():
    pass


def create_valid(dataset, test_len=5000, movie=True):
    dataset = dataset.reset_index()
    ind_exc = np.random.permutation(len(dataset))[0:test_len]
    test_set = dataset.iloc[ind_exc]
    if movie is True:
        dataset.drop(index=ind_exc, inplace=True)
        use_v = test_set['userId'].unique()
        mov_v = test_set['movieId'].unique()
        use_t = dataset['userId'].unique()
        mov_t = dataset['movieId'].unique()
        index_u = list(np.intersect1d(use_v, use_t))
        index_m = list(np.intersect1d(mov_v, mov_t))
        dataset = dataset.pivot(index="movieId", columns="userId", values="rating")
        test_set = test_set.pivot(index="movieId", columns="userId", values="rating")
        dataset = dataset[index_u]
        test_set = test_set.loc[index_m][index_u]
    else:
        dataset.drop(index=ind_exc, inplace=True)
        use_v = test_set['userId'].unique()
        mov_v = test_set['ISBN'].unique()
        use_t = dataset['userId'].unique()
        mov_t = dataset['ISBN'].unique()
        index_u = list(np.intersect1d(use_v, use_t))
        index_m = list(np.intersect1d(mov_v, mov_t))
        dataset = dataset.pivot(index="ISBN", columns="userId", values="rating")
        test_set = test_set.pivot(index="ISBN", columns="userId", values="rating")
        dataset = dataset[index_u]
        test_set = test_set.loc[index_m][index_u]

    return dataset, test_set


def knn_uu_cosine(ratings, k_users, normalization):
    # exclude test set
    df_rating, df_test_set = create_valid(ratings)

    # rating table
    df_rating_raw = df_rating

    # normalization procedure
    if normalization == 'centering + division by variance':
        df_rating = (df_rating - df_rating.mean()) / df_rating.var() ** 0.5
    elif normalization == 'centering':
        df_rating = df_rating - df_rating.mean()
    df_rating = df_rating.fillna(0)
    user_std = (df_rating * df_rating).mean() ** 0.5

    # calc cov matrix
    user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
    user_corr = user_corr.fillna(0)

    # calc errors
    y_pred = []

    for user_number, val in df_test_set.iteritems():
        # index of nearest users
        sorted_index = list(np.argsort(user_corr[user_number]))[::-1]

        # test set movies
        test_mov = val.dropna()
        test_mov_id = list(test_mov.index)
        train_mov_id = list(df_rating.index)
        test_mov_id = list(set(test_mov_id) & set(train_mov_id))
        test_mov = test_mov[test_mov_id]

        # sum of their ratings weighted by the corr
        if len(test_mov) > 0:
            corr_k = user_corr.iloc[sorted_index[1:k_users + 1]][[user_number]].values
            ratings_k = df_rating.loc[test_mov_id].iloc[:, sorted_index[1:k_users + 1]].values
            w_sum_k = ratings_k @ corr_k
            mv_rated = df_rating_raw.loc[test_mov_id].iloc[:, sorted_index[1:k_users + 1]].notnull().values
            seen_sim_len = mv_rated @ corr_k
            seen_sim_len = 1 / (seen_sim_len + (seen_sim_len == 0))
            if normalization == 'centering + division by variance':
                recommended = w_sum_k * seen_sim_len * df_rating_raw[user_number].var() ** 0.5 + df_rating_raw[
                    user_number].mean()
            elif normalization == 'centering':
                recommended = w_sum_k * seen_sim_len + df_rating_raw[user_number].mean()
            y_pred += list(recommended)

    return y_pred, df_test_set


if __name__ == "__main__":
    main()
