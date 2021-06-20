import numpy as np
import pandas as pd
import plotly.graph_objects as go
import copy
from tmdbv3api import TMDb
from tmdbv3api import Movie
from scipy.spatial.distance import hamming, euclidean, chebyshev, cityblock
import urllib.request


def main():
    ratings = pd.read_csv('ratings.csv')
    for i in range(10):
        y_pred, df_test_set = knn_uu_cosine(ratings, 15)
        saving = all_performance_measures(*group_test_results(y_pred, df_test_set))
        print(saving)


def group_test_results(predicted, actuals):
    """groups preditions and actual values"""
    pred_vector = np.hstack(predicted)
    actual_vector = np.hstack(actuals)
    df_actual_pred = pd.DataFrame({'actual': actual_vector, 'predicted': pred_vector}, columns=['actual', 'predicted'])
    groups_by_actual = []
    group_header_actual = []
    groups_by_pred = []
    group_header_pred = []

    for x in np.sort(df_actual_pred.actual.unique()):
        group = df_actual_pred[df_actual_pred["actual"] == x]
        groups_by_actual.append(group)
        group_header_actual.append(x)

    for x in np.sort(df_actual_pred.predicted.unique()):
        group = df_actual_pred[df_actual_pred["predicted"] == x]
        groups_by_pred.append(group)
        group_header_pred.append(x)
    return df_actual_pred, groups_by_actual, group_header_actual, groups_by_pred, group_header_pred


def basic_measures(df, onlypred=False, onlyactual=False):
    pred = df["predicted"]
    actual = df["actual"]
    if (onlypred is False) and (onlyactual is False):
        std_actual = np.nanstd(actual)
        average_actual = np.nanmean(actual)
        average_pred = np.nanmean(pred)
        std_pred = np.nanstd(pred)
        return average_actual, average_pred, std_actual, std_pred
    elif onlyactual is True:
        average_actual = np.nanmean(actual)
        std_actual = np.nanstd(actual)
        return average_actual, std_actual
    elif onlypred is True:
        average_pred = np.nanmean(pred)
        std_pred = np.nanstd(pred)
        return average_pred, std_pred


def mse(df):
    """mean squared error"""
    df = df.dropna()
    actual = df["actual"]
    pred = df["predicted"]
    if len(pred)== 0 or len(actual) == 0:
        return 0
    else:
        return sum((actual - pred) ** 2) * 1 / len(pred)


def rmse(df):
    """root mean squared error"""
    df = df.dropna()
    actual = df["actual"]
    pred = df["predicted"]
    if len(pred)== 0 or len(actual) == 0:
        return 0
    else:
        return sum((actual - pred) ** 2) * 1 / len(pred)


def mae(df):
    """mean absolute error"""
    df = df.dropna()
    actual = df["actual"]
    pred = df["predicted"]
    if len(pred)== 0 or len(actual) == 0:
        return 0
    else:
        return sum((actual - pred) ** 2) * 1 / len(pred)


def all_performance_measures(df_actual_pred, groups_by_actual, group_header_actual, groups_by_pred, group_header_pred):
    """calcualtes all performance measures for a given predict|test dataframe"""
    average_actual1, average_pred1, std_actual1, std_pred1 = basic_measures(df_actual_pred)
    mse1, rmse1, mae1 = mse(df_actual_pred), rmse(df_actual_pred), mae(df_actual_pred)
    df_basic_measures_for_all_testpoints = pd.DataFrame(
        np.array([average_actual1, average_pred1, std_actual1, std_pred1]),
        index=["average_actual", "average_pred", "std_actual", "std_pred"])
    df_performance_for_all_testpoints = pd.DataFrame(np.array([mse1, rmse1, mae1]),
                                                     index=["MSE", "RMSE", "MAE"])
    # iterates through the groups within the actual ratings and
    # calcs performance measures
    c = 0
    df_groups_actual = [] # for testing of heuristik
    plot_groups_actual = []
    for x in groups_by_actual:

        if len(x["predicted"].dropna()) == 0:
            group_header_actual.pop(c)
            continue
        #            print("One group doesnt have predictions: ENDING PROGRAM",x)
        #            quit()
        plot_parameters = []
        plot_parameters.append(group_header_actual[c])
        plot_parameters.append(len(x["predicted"].dropna()))
        average_pred, std_pred = basic_measures(x, onlypred=True)
        plot_parameters.append(average_pred)
        plot_parameters.append(std_pred)
        plot_parameters.append(mse(x))
        plot_parameters.append(rmse(x))
        plot_parameters.append(mae(x))
        if c == 0:
            df_groups_actual = pd.DataFrame(np.array(plot_parameters),
                                            index=["actual rating", "no of predictions", "average", "std", "mse",
                                                   "rmse", "mae"],
                                            columns=[group_header_actual[c]])
        else:
            df_groups_actual[group_header_actual[c]] = np.array(plot_parameters)
        c += 1
        plot_groups_actual.append(copy.deepcopy(plot_parameters))

    # optional not to have too many groups
    # groups_by_pred = list(groups_by_pred[:], groups_by_pred[-3:])
    c = 0
    plot_groups_pred = []
    for x in groups_by_pred:
        plot_parameters = []
        plot_parameters.append(group_header_pred[c])
        if len(x["predicted"].dropna()) == 0:
            group_header_pred.pop(c)
            continue
        plot_parameters.append(len(x["predicted"].dropna()))
        average_actual, std_actual = basic_measures(x, onlyactual=True)
        plot_parameters.append(average_actual)
        plot_parameters.append(std_actual)
        plot_parameters.append(mse(x))
        plot_parameters.append(rmse(x))
        plot_parameters.append(mae(x))
        if c == 0:
            df_groups_pred = pd.DataFrame(np.array(plot_parameters),
                                          index=["predicted rating group", "no of predictions in group",
                                                 "average", "std", "mse", "rmse", "mae"],
                                          columns=[group_header_pred[c]])
        else:
            df_groups_pred[group_header_pred[c]] = np.array(plot_parameters)
        c += 1
        plot_groups_pred.append(copy.deepcopy(plot_parameters))
    return df_basic_measures_for_all_testpoints, df_performance_for_all_testpoints, df_groups_actual, df_groups_pred


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


def knn_uu_cosine(ratings, k_users):
    # exclude test set
    df_rating, df_test_set = create_valid(ratings)

    # rating table
    df_rating_raw = df_rating

    # normalization procedure
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
            recommended = w_sum_k * seen_sim_len + df_rating_raw[user_number].mean()
            y_pred += list(recommended)

    return np.array(y_pred).T, df_test_set.T.stack().values


if __name__ == "__main__":
    main()
