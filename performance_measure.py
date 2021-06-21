import numpy as np
import pandas as pd
import copy
from scipy.spatial.distance import hamming, euclidean, chebyshev, cityblock


def get_book_data(filter_tr, like_to_value=True):
    # load data from csv
    books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                     'imageUrlL']

    users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    users.columns = ["userId", "location", "age"]

    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings.columns = ["userId", "ISBN", "rating"]

    ratings = ratings.drop_duplicates(subset=["userId", "ISBN"])

    # filter dataset for users / items with much interaction
    u = ratings.userId.value_counts()
    b = ratings.ISBN.value_counts()

    ratings = ratings[ratings.userId.isin(u.index[u.gt(filter_tr)])]
    ratings = ratings[ratings.ISBN.isin(b.index[b.gt(filter_tr)])]

    # create table
    df_ratings = ratings.pivot(index="ISBN", columns="userId", values="rating")
    df_rating_nonzero = ratings.loc[ratings["rating"].values != 0]
    if like_to_value:
        percentiles = df_ratings.describe(include='all').iloc[6].values
        df_zeros = df_ratings.values == 0
        df_ratings = df_ratings + df_zeros * percentiles
        df_zeros = df_ratings == 0
        df_ratings = df_ratings + df_zeros * np.mean(percentiles[percentiles != 0])
        ratings["rating"] = df_ratings.stack().values

    # second output has the raw data format
    return df_ratings, ratings, df_rating_nonzero, books, users


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

## get sim and predictions for distances != cosine
def similarity_calculation_distances(df_rating, distance_measure, user_number):
    distances = []
    similarities = []
    for x in range(df_rating.shape[1]):
        if distance_measure == "euclidean":
            dist = euclidean(df_rating.loc[:, user_number], df_rating.iloc[:, x])
            distances.append(dist)
            similarities.append(1 / (1 + dist))
        elif distance_measure == "manhattan (city block)":
            dist = cityblock(df_rating.loc[:, user_number], df_rating.iloc[:, x])
            distances.append(dist)
            similarities.append(1 / (1 + dist))
        elif distance_measure == "hamming":
            dist = hamming(df_rating.loc[:, user_number], df_rating.iloc[:, x])
            distances.append(dist)
            similarities.append(1 - dist)
        elif distance_measure == "chebyshev":
            dist = chebyshev(df_rating.loc[:, user_number], df_rating.iloc[:, x])
            distances.append(dist)
            similarities.append(1 / (1 + dist))
    return similarities


def predicted_ratings_distances(df_rating, similarities, user_number, k_users, df_rating_raw
                                , replacenan=False, replacement=0, weigthing=False, testing=False, weighting=True):
    sorted_index = list(np.argsort(similarities))[::-1][1:k_users + 1]
    rated = df_rating.iloc[:, sorted_index]
    if testing is False: rated = rated[df_rating_raw[user_number].isnull()]
    if weighting is False:
        predicted_ratings = rated.mean(axis=1)
    else:
        sim_array = np.array(similarities[1:k_users + 1])
        sim_array = np.tile(sim_array, (rated.shape[0], 1))
        sim_array = sim_array * rated.notna()
        predicted_ratings = np.nansum(rated * sim_array, axis=1)
        predicted_ratings = predicted_ratings / np.nansum(sim_array, axis=1)
        predicted_ratings = pd.Series(predicted_ratings, index=rated.index)
    if replacenan is True: predicted_ratings.fillna(replacement, inplace=True)
    return predicted_ratings


def create_corr_matrix(df_rating, normalization='centering'):
    """creates a correlation matrix over the items"""
    df_rating = df_rating.transpose()
    if normalization == 'centering':
        df_rating = df_rating - df_rating.mean()
        df_rating = df_rating.fillna(0)
    elif normalization == 'centering + division by variance':
        'centering + division by variance'
    elif normalization == "None":
        df_rating = df_rating.fillna(df_rating.mean())

    df_rating = df_rating.transpose()

    corr_movies = np.corrcoef(df_rating)
    corr_movies = pd.DataFrame(corr_movies)
    corr_movies = corr_movies.set_index(df_rating.index)
    corr_movies.columns = df_rating.index
    return corr_movies


def item_item_cf(df_rating, corr_matrix, user_number, k_items=15, test_labels=[], weighting=True):
    """returns predictions based on item item cf"""
    if len(test_labels) == 0:
        testing = False
    else:
        testing = True

    k_items_original = k_items
    corr_matrix_raw = corr_matrix.copy()
    corr_matrix = corr_matrix.fillna(-20)

    # rating table
    df_rating_raw = df_rating

    if testing is False:
        user_items = df_rating_raw[user_number]
    else:
        user_items = df_rating_raw[user_number][test_labels]
    item_indices = list(user_items.index)
    counter = 0
    predictions = []
    for x in user_items:
        k_items = k_items_original
        k_items += 1
        if x != x:
            # find the index of the k most similar movies within the correlation matrix
            current_movie_id = item_indices[counter]
            if corr_matrix_raw[current_movie_id].isnull().all():
                counter += 1
                predictions.append(np.nan)
            else:
                k_most_similar = np.argpartition(corr_matrix[current_movie_id], -k_items)
                if -20 in np.partition(corr_matrix[current_movie_id], -k_items)[-k_items:]:
                    k_items_reduced = len([x for x in k_items if x != -20])
                    k_most_similar = k_most_similar[-k_items_reduced:]
                else:
                    k_most_similar = k_most_similar[-k_items:]

                # predict based on the average of the user for the k movies
                #  print(df_rating_raw[user_number].iloc[k_most_similar])
                # isnull().all()
                # if all movies were not seen by the user append nan
                if np.mean(df_rating_raw[user_number].iloc[k_most_similar]) != np.mean(
                        df_rating_raw[user_number].iloc[k_most_similar]):
                    predictions.append(np.nan)
                # else append the average
                else:

                    corrs = np.partition(corr_matrix[current_movie_id], -k_items)[-k_items:]
                    df_rating_corr = df_rating_raw[user_number]
                    if weighting is False:
                        predictions.append(np.nanmean(df_rating_raw[user_number].iloc[k_most_similar]))

                    else:
                        predictions.append(np.nansum(df_rating_corr.iloc[k_most_similar] * corrs) / (
                            np.sum(corrs[df_rating_corr.iloc[k_most_similar].notnull()])))
                        a = df_rating_corr
                        b = df_rating_corr.iloc[k_most_similar]
                        c = np.sum(corrs[df_rating_corr.iloc[k_most_similar].notnull()])
                        d = df_rating_corr.iloc[k_most_similar] * corrs
                        pass

                counter += 1

        # nan is not equal to itself, so if movie is not seen
        # the entry is nan
        else:
            predictions.append(-50)
            counter += 1
    predictions = pd.Series(predictions, index=item_indices)
    return predictions


# performance measures
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
        return np.sqrt(sum((actual - pred) ** 2) * 1 / len(pred))


def mae(df):
    """mean absolute error"""
    df = df.dropna()
    actual = df["actual"]
    pred = df["predicted"]
    if len(pred)== 0 or len(actual) == 0:
        return 0
    else:
        return sum(np.abs(actual - pred)) * 1 / len(pred)


def test_generation_distances(ratings, movie=True):

    #### hyperparameters
    k_users = 23
    normalization = 'centering + division by variance'
    distance_measure = "euclidean"
    ####

    train, test = create_valid(ratings,5000 ,movie)
    df_rating = train
    df_rating_raw = df_rating.copy()
    if normalization == 'centering + division by variance':
        df_rating = (df_rating - df_rating.mean()) / df_rating.var() ** 0.5
        df_rating = df_rating.fillna(0)
    elif normalization == 'centering':
        df_rating = df_rating - df_rating.mean()
        df_rating = df_rating.fillna(0)
    elif normalization == "None":
        df_rating = df_rating.fillna(df_rating.mean())
    ###### the part ends here
    c = 0
    predicted = []
    actuals = []
    for x in test:
        # check if any value for a user is in test set
        if c % 10 == 0: print(c)
        if test[x].notna().values.any():
            similarities = similarity_calculation_distances(df_rating, distance_measure, x)
            user_average = df_rating_raw[x].mean()

            # change replacenan to true to replace nans with user average
            predicted_ratings = predicted_ratings_distances(df_rating_raw, similarities, x, k_users,
                                                            df_rating_raw, replacenan=False, replacement=user_average,
                                                            testing=True)
            index = list(test[x].dropna().index)
            actual = test[x].dropna()
            predicted.append(predicted_ratings[predicted_ratings.index.isin(index)])
            actuals.append(actual)
            c += 1
        else:
            pass
    return predicted, actuals


def test_generation_item_cf(ratings, movie=True):
    """generates predictions for a test set with item-item cf"""

    train, test = create_valid(ratings, 5000, movie)
    df_rating = train
    df_rating_raw = df_rating.copy()
    c = 0
    user_number = 1
    predicted = []
    actuals = []
    corr_matrix = create_corr_matrix(df_rating_raw)
    for x in test:
        if c % 10 == 0:
            print(c)
        # check if any value for a user is in test set
        if test[x].notna().values.any():
            predicted_ratings = item_item_cf(df_rating, corr_matrix, x, 15, test[x].dropna().index)
            index = list(test[x].dropna().index)
            actual = test[x].dropna()
            predicted.append(predicted_ratings[predicted_ratings.index.isin(index)])
            actuals.append(actual)
            c += 1
            ###
        else:
            pass
    return predicted, actuals


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
                                            index=["actual rating", "no of actual ratings", "average", "std", "mse",
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


def item_item_cf_heuristik(df_rating, user_number=69, neighbours = 15, no_similar_to_favorite = 5, no_of_recommendations= 3, corr_matrix = False):
    favorites = df_rating[user_number][df_rating[user_number] == df_rating[user_number].max()].index
    favorites = np.random.permutation(favorites)

    ### replace problematic values in corr matrix
    if corr_matrix is False:
        corr_matrix = create_corr_matrix(df_rating)
    else:
        corr_matrix = corr_matrix
    corr_matrix = corr_matrix.fillna(-20)

    corr_matrix_reduced = corr_matrix.copy()
    corr_matrix = corr_matrix.drop(list(df_rating[user_number].dropna().index), axis=0)
    #corr_matrix_reduced = corr_matrix_reduced.drop(favorites, axis=0)
    corr_matrix_reduced = corr_matrix_reduced.drop(favorites, axis=1)
    most_correlated = []
    correlation_of_most_correlated = []
    already_removed = []
    for x in favorites:
        k = np.argpartition(corr_matrix[x], -no_similar_to_favorite)[-no_similar_to_favorite:]
        similar_to_favorite = corr_matrix[x].iloc[k].index
        most_correlated.append(similar_to_favorite)
        already_removed.extend(similar_to_favorite)
        for y in similar_to_favorite:
            if y in already_removed:
                pass
            else:
                corr_matrix_reduced = corr_matrix_reduced.drop(y, axis=0)
    corr_matrix_local = corr_matrix_reduced
    similar_to_favorites_rated = []
    c = 0
    for x in most_correlated:
        l = []
        for y in x:
            #corr_matrix_local = corr_matrix_reduced.drop(favorites[c], axis=0)
            h = np.argpartition(corr_matrix_local[y], -neighbours)[-neighbours:]
            correlations_to_neighbours = np.partition(corr_matrix_local[y], -neighbours)[-neighbours:]
            index_neighbours = corr_matrix_local[y].iloc[h].index
            if favorites[c] in index_neighbours:
                index_neighbours = index_neighbours.drop(favorites[c])
            rating = np.nanmean(df_rating[user_number][index_neighbours])
            l.append(np.nanmean(df_rating[user_number][index_neighbours]))
            # get neighbours most correlated keep correlation
            # calc mean among them in df_rating weighted by correlation
            # save results
        similar_to_favorites_rated.append(l)
        c +=1
    df_similar_to_favorites_rated = pd.DataFrame(np.array(similar_to_favorites_rated).T, columns = favorites)
    idx = np.random.permutation(df_similar_to_favorites_rated.index)
    df_names = pd.DataFrame(np.array(most_correlated).T, columns = favorites)
    df_similar_to_favorites_rated = df_similar_to_favorites_rated.reindex(idx)
    df_names = df_names.reindex(idx)
    ratings = df_similar_to_favorites_rated.fillna(-10).stack().nlargest(3)
    index = list(ratings.index)#df_similar_to_favorites_rated.max().fillna(-10).nlargest(no_of_recommendations).index
    for x in range(len(index)):
        index[x] = index[x][1]
    max_index = df_similar_to_favorites_rated.loc[:,index].idxmax()
    best_movies = []
    c = 0
    for x in index:
        try:
            best_movies.append(df_names[x][list(max_index)[c]])
            c += 1
        except:
            best_movies.append(np.nan)
            c += 1
    ratings.index = best_movies

    ratings = ratings[~ratings.index.duplicated(keep='first')]
    return ratings, index, best_movies


def performance_item_item_cf(ratings, movie=True):
    """performance for item-item cf"""
    pred, actuals = test_generation_item_cf(ratings, movie)
    g = group_test_results(pred, actuals)
    results = all_performance_measures(*g)
    return results


def performance_user_user_cf_distances(ratings, movie=True):
    """performance for user user cf with distance != cosine"""
    pred, actuals = test_generation_distances(ratings, movie)
    g = group_test_results(pred, actuals)
    results = all_performance_measures(*g)
    return results


def all_performances(movie= True, filter_tr = 20):
    if movie is True:
        rating = pd.read_csv('ratings.csv')
        result_item = performance_item_item_cf(rating.copy())
        result_distance = performance_user_user_cf_distances(rating.copy())
        return  result_item, result_distance

    else:
        rating = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
        rating.columns = ["userId", "ISBN", "rating"]

        rating = get_book_data(filter_tr)[1]
        result_item = performance_item_item_cf(rating.copy(), movie=False)
        result_distance = performance_user_user_cf_distances(rating.copy(), movie=False)
    return result_item, result_distance


def many_predictions(no_predictions=1):

    item_movie, distance_movie  = all_performances()
    item_book, distance_book = all_performances(False)

    item_movie_basic = item_movie[0]
    item_movie_errors_total = item_movie[1]
    item_movie_group_actual = item_movie[2]
    item_movie_group_predicted = item_movie[3]

    distance_movie_basic = distance_movie[0]
    distance_movie_errors_total = distance_movie[1]
    distance_movie_group_actual = distance_movie[2]
    distance_movie_group_predicted = distance_movie[3]

    item_book_basic = item_book[0]
    item_book_errors_total = item_book[1]
    item_book_group_actual = item_book[2]
    item_book_group_predicted = item_book[3]

    distance_book_basic = distance_book[0]
    distance_book_errors_total = distance_book[1]
    distance_book_group_actual = distance_book[2]
    distance_book_group_predicted = distance_book[3]
    for x in range(no_predictions-1):
        item_movie, distance_movie = all_performances()
        item_book, distance_book = all_performances(False)

        item_movie_basic += item_movie[0]
        item_movie_errors_total += item_movie[1]
        item_movie_group_actual += item_movie[2]
        item_movie_group_predicted += item_movie[3]

        distance_movie_basic += distance_movie[0]
        distance_movie_errors_total += distance_movie[1]
        distance_movie_group_actual += distance_movie[2]
        distance_movie_group_predicted += distance_movie[3]

        item_book_basic += item_book[0]
        item_book_errors_total += item_book[1]
        item_book_group_actual += item_book[2]
        item_book_group_predicted += item_book[3]

        distance_book_basic += distance_book[0]
        distance_book_errors_total += distance_book[1]
        distance_book_group_actual += distance_book[2]
        distance_book_group_predicted += distance_book[3]


        item_movie_basic = item_movie_basic / no_predictions
        item_movie_errors_total = item_movie_errors_total / no_predictions
        item_movie_group_actual = item_movie_group_actual / no_predictions
        item_movie_group_predicted = item_movie_group_predicted / no_predictions

        distance_movie_basic = distance_movie_basic / no_predictions
        distance_movie_errors_total = distance_movie_errors_total / no_predictions
        distance_movie_group_actual = distance_movie_group_actual / no_predictions
        distance_movie_group_predicted = distance_movie_group_predicted / no_predictions

        item_book_basic = item_book_basic / no_predictions
        item_book_errors_total = item_book_errors_total / no_predictions
        item_book_group_actual = item_book_group_actual / no_predictions
        item_book_group_predicted = item_book_group_predicted / no_predictions

        distance_book_basic = distance_book_basic / no_predictions
        distance_book_errors_total = distance_book_errors_total / no_predictions
        distance_book_group_actual = distance_book_group_actual / no_predictions
        distance_book_group_predicted = distance_book_group_predicted / no_predictions
    x = 0
    item_movie_basic.to_csv("perf/" + str(x) + "item_movie"+"_basic" + "csv")
    item_movie_errors_total.to_csv("perf/" + str(x) + "item_movie"+"_errors_total" + "csv")
    item_movie_group_actual.to_csv("perf/" + str(x) + "item_movie"+"_group_actual" + "csv")
    item_movie_group_predicted.to_csv("perf/" + str(x) + "item_movie"+"_group_predicted" + "csv")

    distance_movie_basic.to_csv("perf/" + str(x) + "distance_movie"+"_basic" + "csv")
    distance_movie_errors_total.to_csv("perf/" + str(x) + "distance_movie"+"_errors_total" + "csv")
    distance_movie_group_actual.to_csv("perf/" + str(x) + "distance_movie"+"_group_actual" + "csv")
    distance_movie_group_predicted.to_csv("perf/" + str(x) + "distance_movie"+"_group_predicted" + "csv")

    item_book_basic.to_csv("perf/" + str(x) + "item_book"+"_basic" + "csv")
    item_book_errors_total.to_csv("perf/" + str(x) + "item_book"+"_errors_total" + "csv")
    item_book_group_actual.to_csv("perf/" + str(x) + "item_book"+"_group_actual" + "csv")
    item_book_group_predicted.to_csv("perf/" + str(x) + "item_book"+"_group_predicted" + "csv")

    distance_book_basic.to_csv("perf/" + str(x) + "distance_book"+"_basic" + "csv")
    distance_book_errors_total.to_csv("perf/" + str(x) + "distance_book"+"_errors_total" + "csv")
    distance_book_group_actual.to_csv("perf/" + str(x) + "distance_book"+"_group_actual" + "csv")
    distance_book_group_predicted.to_csv("perf/" + str(x) + "distance_book"+"_group_predicted" + "csv")

many_predictions()