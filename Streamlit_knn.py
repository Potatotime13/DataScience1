import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import copy
from tmdbv3api import TMDb
from tmdbv3api import Movie
from scipy.spatial.distance import hamming, euclidean, chebyshev, cityblock
import urllib.request


def main():
    st.title('Data Science: Recommender Systems')
    with st.sidebar:
        st.write('Dataset selection')

    c_task = st.sidebar.selectbox(
        "",
        ("MovieLens", "MovieLens-Tech", "Books")
    )
    if c_task == "MovieLens":
        task1()
    if c_task == "MovieLens-Tech":
        task2()
    if c_task == "Books":
        task3()
    if c_task == "Books-Tech":
        task4()


def movie_url(ids):
    tmdb = TMDb()
    tmdb.api_key = '52f358adec9f89bb2d9a47fceda64fdc'
    tmdb.language = 'en'
    tmdb.debug = True
    movie_api = Movie()
    urls = []
    info = []
    for m_id in ids:
        m = movie_api.details(int(m_id[0]))
        urls.append(m.poster_path)
        info.append(m.title)
    return urls, info


def cosine(df_rating, user_number, k_users, df_rating_raw, normalization):
    user_std = (df_rating * df_rating).mean() ** 0.5

    # calc cov matrix
    user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
    user_corr = user_corr.fillna(0)

    # index of nearest users
    sorted_index = list(np.argsort(user_corr[user_number]))[::-1]

    # sum of their ratings weighted by the corr
    corr_k = user_corr.iloc[sorted_index[1:k_users + 1]][[user_number]].values
    ratings_k = df_rating.iloc[:, sorted_index[1:k_users + 1]].values
    w_sum_k = ratings_k @ corr_k
    mv_rated = df_rating_raw.iloc[:, sorted_index[1:k_users + 1]].notnull().values
    seen_sim_len = mv_rated @ corr_k
    seen_sim_len = 1 / (seen_sim_len + (seen_sim_len == 0))
    recommended = w_sum_k * seen_sim_len
    if normalization == 'centering + division by variance':
        recommended = w_sum_k * seen_sim_len * df_rating_raw[user_number].var() ** 0.5 + df_rating_raw[
            user_number].mean()
    elif normalization == 'centering':
        recommended = w_sum_k * seen_sim_len + df_rating_raw[user_number].mean()
    elif normalization == 'None':
        recommended = w_sum_k * seen_sim_len
    unseen = df_rating_raw[user_number].isnull().values
    recommended = recommended.T[0] * unseen
    return recommended


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
                                ,replacenan=False,replacement = 0, weigthing=False, testing=False):
    sorted_index = list(np.argsort(similarities))[::-1][1:k_users + 1]
    # w_sum_k = rating_k * weighting vector (abhängig von sim!)
    rated = df_rating.iloc[:, sorted_index]
    if testing is False: rated = rated[df_rating_raw[user_number].isnull()]

    predicted_ratings = rated.mean(axis=1)
    if replacenan is True: predicted_ratings.fillna(replacement, inplace=True)
    return predicted_ratings

def create_corr_matrix(df_rating, normalization='centering' ):
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


def item_item_cf(df_rating, corr_matrix,user_number, k_items=10, test_labels = [], weighting=True):
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
    df_rating = df_rating.fillna(df_rating.mean(axis=1))
    df_rating = df_rating.transpose()
    if testing is False:
        user_items = df_rating_raw[user_number]
    else:
        user_items = df_rating_raw[user_number][test_labels]
    item_indices = list(user_items.index)
    counter = 0
    predictions = []
    for x in user_items:
        if counter % 100 == 0: print(counter)
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
                        predictions.append(np.nansum(df_rating_corr.iloc[k_most_similar] * corrs)/(np.sum(corrs[df_rating_corr.iloc[k_most_similar].notnull()])))
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




#rating = pd.read_csv('ratings.csv')
#df_rating = rating.pivot(index="movieId", columns="userId", values="rating")
#movies = pd.read_csv('movies.csv')
#df_rating_raw = df_rating.copy()
#corr_matrix = create_corr_matrix(df_rating_raw)
#predicted_ratings = item_item_cf(df_rating, corr_matrix, 15, 20) ##for movies replace 79186 with i e [1:610]
#print()






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
        std_actual= np.nanstd(actual)
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
    return sum((actual - pred) ** 2) * 1 / len(pred)


def rmse(df):
    """root mean squared error"""
    df = df.dropna()
    actual = df["actual"]
    pred = df["predicted"]
    return np.sqrt(sum((actual - pred) ** 2) * 1 / len(pred))


def mae(df):
    """mean absolute error"""
    df = df.dropna()
    actual = df["actual"]
    pred = df["predicted"]
    return sum(np.abs(actual - pred)) * 1 / len(pred)


def create_valid(dataset, test_len=5000, movie=True):
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


def color_descends(rec):
    color_grade = rec + abs(rec.min())
    if rec.max() + abs(rec.min()) > 0:
        color_grade = color_grade / (rec.max() + abs(rec.min()))
    else:
        color_grade *= 1
    color_grade = np.sort(np.array(color_grade))
    color_grade = np.flip(color_grade)

    colors = []
    for percentage in color_grade:
        colors.append('rgba(255,185,15,' + str(round(percentage ** 2, 3)) + ')')
    return colors


def test_generation_distances(ratings, movie=True):
    ##### this part possibly in own function


    user_number = st.sidebar.selectbox("User ID", (10, 12, 69, 52, 153))
    k_users = st.sidebar.selectbox("K nearest", (23, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))
    normalization = st.sidebar.selectbox("Normalization",
                                         ('centering + division by variance', 'centering', "None"))
    distance_measure = st.sidebar.selectbox("distance_measure",
                                            ("euclidean", 'cosine', "euclidean", "manhattan (city block)", "hamming",
                                             "chebyshev"))
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

        print(c)
        if test[x].notna().values.any():
            similarities = similarity_calculation_distances(df_rating, distance_measure, x)
            user_average = df_rating_raw[user_number].mean()
            # change replacenan to true to replace nans with user average
            predicted_ratings = predicted_ratings_distances(df_rating_raw, similarities, user_number, k_users,
                                                            df_rating_raw, replacenan=False, replacement=user_average, testing=True)
            index = list(test[x].dropna().index)
            actual = test[x].dropna()
            predicted.append(predicted_ratings[predicted_ratings.index.isin(index)])
            actuals.append(actual)
            c += 1
        else:
            pass
    return predicted, actuals


def test_generation_item_cf(ratings, movie = True):
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
        if c % 10 ==0:
            print(c)
        if c == 100:
            break
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
    sorted_values = np.argsort(actual_vector)
    df_actual_pred = pd.DataFrame({'actual': actual_vector , 'predicted': pred_vector}, columns=['actual', 'predicted'])
    groups_by_actual = []
    group_header_actual = []
    groups_by_pred = []
    group_header_pred = []

    for x in np.sort(df_actual_pred.actual.unique()):
        group = df_actual_pred[df_actual_pred["actual"]==x]
        groups_by_actual.append(group)
        group_header_actual.append(x)

### Dummer Film Problem -.-
    for x in np.sort(df_actual_pred.predicted.unique()):
        group = df_actual_pred[df_actual_pred["predicted"]==x]
        groups_by_pred.append(group)
        group_header_pred.append(x)
    return df_actual_pred, groups_by_actual, group_header_actual, groups_by_pred,  group_header_pred


def all_performance_measures(df_actual_pred, groups_by_actual, group_header_actual, groups_by_pred, group_header_pred):
    """calcualtes all performance measures for a given predict|test dataframe"""
    average_actual1, average_pred1, std_actual1, std_pred1 = basic_measures(df_actual_pred)
    mse1, rmse1, mae1 = mse(df_actual_pred), rmse(df_actual_pred), mae(df_actual_pred)
    df_basic_measures_for_all_testpoints = pd.DataFrame(np.array([average_actual1, average_pred1, std_actual1, std_pred1]),
                            index= ["average_actual", "average_pred", "std_actual", "std_pred"])
    df_performance_for_all_testpoints = pd.DataFrame(np.array([mse1, rmse1, mae1]),
                            index= ["MSE","RMSE","MAE"])
    # iterates through the groups within the actual ratings and
    # calcs performance measures
    c = 0
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
            df_groups_actual = pd.DataFrame(np.array(plot_parameters), index=["actual rating","no of predictions","average", "std", "mse", "rmse", "mae"],
                                     columns=[group_header_actual[c]])
        else:
            df_groups_actual[group_header_actual[c]] = np.array(plot_parameters)
        c += 1
        plot_groups_actual.append(copy.deepcopy(plot_parameters))

    # optional not to have too many groups
    #groups_by_pred = list(groups_by_pred[:], groups_by_pred[-3:])
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
            df_groups_pred = pd.DataFrame(np.array(plot_parameters), index=["predicted rating group","no of predictions in group",
                                                                            "average", "std", "mse", "rmse", "mae"],
                                                                        columns=[group_header_pred[c]])
        else:
            df_groups_pred[group_header_pred[c]] = np.array(plot_parameters)
        c += 1
        plot_groups_pred.append(copy.deepcopy(plot_parameters))
    return df_basic_measures_for_all_testpoints, df_performance_for_all_testpoints, df_groups_actual, df_groups_pred


def item_item_cf_heuristik(df_rating, user_number=69, neighbours = 15, no_similar_to_favorite = 5, no_of_recommendations= 3):
    favorites = df_rating[user_number][df_rating[user_number] == df_rating[user_number].max()].index
    ### replace problematic values in corr matrix

    corr_matrix = create_corr_matrix(df_rating)
    corr_matrix = corr_matrix.fillna(-20)

    corr_matrix_reduced = corr_matrix.copy()
    corr_matrix = corr_matrix.drop(list(df_rating[user_number].dropna().index), axis=0)
    corr_matrix_reduced = corr_matrix_reduced.drop(favorites, axis=0)
    corr_matrix_reduced = corr_matrix_reduced.drop(favorites, axis=1)
    most_correlated = []
    correlation_of_most_correlated = []
    already_removed = []
    for x in favorites:
        k = np.argpartition(corr_matrix[x],-no_similar_to_favorite)[-no_similar_to_favorite:]
        similar_to_favorite = corr_matrix[x].iloc[k].index
        most_correlated.append(similar_to_favorite)
        already_removed.extend(similar_to_favorite)
        for y in similar_to_favorite:
            if y in already_removed:
                pass
            else:
                corr_matrix_reduced = corr_matrix_reduced.drop(y, axis=0)

    similar_to_favorites_rated = []
    for x in most_correlated:
        l = []
        for y in x:
            h = np.argpartition(corr_matrix_reduced[y], -neighbours)[-neighbours:]
            correlations_to_neighbours = np.partition(corr_matrix_reduced[y], -neighbours)[-neighbours:]
            index_neighbours = corr_matrix_reduced[y].iloc[h].index
            l.append(np.nanmean(df_rating[user_number][index_neighbours]))
            # get neighbours most correlated keep correlation
            # calc mean among them in df_rating weighted by correlation
            # save results
        similar_to_favorites_rated.append(l)
    df_similar_to_favorites_rated = pd.DataFrame(np.array(similar_to_favorites_rated).T, columns = favorites)
    df_names = pd.DataFrame(np.array(most_correlated).T, columns = favorites)
    ratings = df_similar_to_favorites_rated.max().fillna(-10).nlargest(no_of_recommendations)
    index = df_similar_to_favorites_rated.max().fillna(-10).nlargest(no_of_recommendations).index
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
    ratings.reindex(best_movies)

    return best_movies, index, ratings



#ratings = pd.read_csv('ratings.csv')
#df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
#print(item_item_cf_heuristik(df_rating))


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


def get_favorite_movies(ratings, user_number):
    # get movies seen by the user
    df_seen = ratings.loc[:, user_number].replace(0, np.nan)
    df_seen = df_seen.dropna(how="all", axis=0)
    # prints a sorted list of the users movies
    # TODO ausgabe wird später als streamlit list erfolgen
    # print("already seen:",df_seen.sort_values(ascending=True))
    return df_seen

def get_items_item_item_cf(item_list,predicted_ratings, list_len, movies = True, na_filler = 0):
    predicted_ratings.fillna(na_filler, inplace=True)
    sorted = np.argsort(predicted_ratings)#[::-1]
    if movies is True:
        output = item_list.iloc[sorted[0:list_len]][['title', 'genres']]
    else:
        output = item_list.iloc[sorted[0:list_len]][['bookTitle', 'bookAuthor']]
    return output


def task1():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')
    st.write('K nearest neighbor')

    # get settings from sidebar
    user_number = st.sidebar.selectbox("User ID", (10, 12, 69, 52, 153))
    k_users = st.sidebar.selectbox("K nearest", (5, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))
    normalization = st.sidebar.selectbox("Normalization",
                                         ('centering + division by variance', 'centering', "None"))
    distance_measure = st.sidebar.selectbox("distance_measure",
                                            ('cosine', "euclidean", "manhattan (city block)", "hamming",
                                             "chebyshev"))
    # split the genres per movie
    movies["genres"] = movies["genres"].str.split('|')

    # rating table
    df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
    df_rating_raw = df_rating

    # normalization procedure
    if normalization == 'centering + division by variance':
        df_rating = (df_rating - df_rating.mean()) / df_rating.var() ** 0.5
        df_rating = df_rating.fillna(0)
    elif normalization == 'centering':
        df_rating = df_rating - df_rating.mean()
        df_rating = df_rating.fillna(0)
    elif normalization == "None":
        df_rating = df_rating.fillna(df_rating.mean())

    # distance calculation and prediction
    if distance_measure == "cosine":
        recommended = cosine(df_rating, user_number, k_users, df_rating_raw, normalization)
        rec = recommended.copy()
        sorted_mov = list(np.argsort(recommended))[::-1]
        output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]
        out2 = recommended[sorted_mov[0:list_len]]

    else:
        similarities = similarity_calculation_distances(df_rating, distance_measure, user_number)
        user_average = df_rating_raw[user_number].mean()
        predicted_ratings = predicted_ratings_distances(df_rating_raw, similarities, user_number, k_users,
                                                        df_rating_raw, replacenan=True, replacement=user_average)
        recommended = predicted_ratings.copy()
        rec = recommended.copy()
        sorted_mov = list(np.argsort(predicted_ratings))[::-1]
        output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]
        out2 = predicted_ratings.sort_values(ascending=False)[0:list_len]
        print(output)
        print()
    # display results
    rec_header = list(output.columns)
    rec_header.insert(0, 'predict')

    layout = go.Layout(
        margin=dict(r=1, l=1, b=20, t=20))

    fig = go.Figure(data=[go.Table(
        columnwidth=[100, 300, 300],
        header=dict(values=rec_header,
                    line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
                    fill_color=['rgb(14, 17, 23)', 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                    align='center', font=dict(color='white', size=20), height=50
                    ),
        cells=dict(values=[np.round(out2, 2), output.title, output.genres],
                   line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
                   fill_color=[np.array(color_descends(rec)), 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                   align='center', font=dict(color='white', size=14), height=30
                   ))
    ], layout=layout)

    # get movie info / covers
    url, info = movie_url(links.iloc[sorted_mov[0:3]][['tmdbId']].values)

    st.write('Deine Top auswahl')

    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)

    col1.header(info[0])
    col4.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[0])
    col2.header(info[1])
    col5.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[1])
    col3.header(info[2])
    col6.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[2])

    st.write('Alle Empfehlungen für dich:')
    st.write(fig)


def task2():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')
    st.write('K nearest neighbor - performance measures')

    # get settings from sidebar
    k_users = st.sidebar.selectbox("K nearest", (15, 20))
    normalization = st.sidebar.selectbox("Normalization", ('centering', 'centering + division by variance'))

    # exclude test set
    df_rating, df_test_set = create_valid(ratings)

    with st.beta_expander("display code"):
        with st.echo('below'):
            # split the genres per movie
            movies["genres"] = movies["genres"].str.split('|')

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

    st.write("average error of a random test set containing 5000 data points:")
    y_pred = np.array(y_pred)
    y_act = df_test_set.stack().values
    error = np.mean((y_pred-y_act)**2)
    st.write(error)

    if False:
        # surface plot
        sorted_index = pd.DataFrame(np.argsort(user_corr.values))
        a = sorted_index.iloc[:, [608, 609]]
        num_cla = 25
        largest = []
        for i in range(a.shape[0]):
            largest.append(user_corr.iloc[a.iloc[i, 0], a.iloc[i, 1]])
        top = a.iloc[(list(np.argsort(largest))[::-1])[0:num_cla]].index

        classes = [[] for i_1 in range(num_cla)]

        for i_2 in range(user_corr.shape[0]):
            sims = user_corr.iloc[[i_2], top].values
            j = list(np.argsort(sims))[::-1]
            classes[j[0][0]].append(int(i_2))
        final_index = []
        for cl in classes:
            final_index += cl
        plot_surf = user_corr.iloc[final_index, final_index]

        fig = go.Figure(data=[go.Surface(z=plot_surf)])
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                          highlightcolor="limegreen", project_z=True))
        fig.update_layout(title='Correlation surface', autosize=True,
                          width=800, height=700,
                          margin=dict(l=1, r=1, b=40, t=40)
                          )

        st.write(fig)


def get_book_data(filter_tr):
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

    return df_ratings, ratings, books, users


def task3():
    st.write('K nearest neighbor')

    # load data
    df_rating, ratings, books, users = get_book_data(200)

    # get settings from sidebar
    user_number = st.sidebar.selectbox("User ID", (79186, 207782))
    k_users = st.sidebar.selectbox("K nearest", (5, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))
    normalization = st.sidebar.selectbox("Normalization",
                                         ('centering + division by variance', 'centering', "None"))
    distance_measure = st.sidebar.selectbox("distance_measure",
                                            ("euclidean",'cosine', "euclidean", "manhattan (city block)", "hamming",
                                             "chebyshev"))

    df_rating_raw = df_rating

    # normalization procedure
    if normalization == 'centering + division by variance':
        df_rating = (df_rating - df_rating.mean()) / df_rating.var() ** 0.5
        df_rating = df_rating.fillna(0)
    elif normalization == 'centering':
        df_rating = df_rating - df_rating.mean()
        df_rating = df_rating.fillna(0)
    elif normalization == "None":
        df_rating = df_rating.fillna(df_rating.mean())

    if distance_measure == "cosine":
        recommended = cosine(df_rating, user_number, k_users, df_rating_raw, normalization)
        rec = recommended.copy()  # recommended books
        sorted_bok = list(np.argsort(recommended))[::-1]
        output = books.iloc[sorted_bok[0:list_len]][['bookTitle', 'bookAuthor']]
        out2 = recommended[sorted_bok[0:list_len]]

    else:
        similarities = similarity_calculation_distances(df_rating, distance_measure, user_number)
        df_rating_mean = df_rating_raw.fillna(df_rating_raw.mean())
        predicted_ratings = predicted_ratings_distances(df_rating_mean, similarities, user_number, k_users,
                                                        df_rating_raw)
        recommended = predicted_ratings.copy()
        rec = recommended.copy()
        sorted_bok = list(np.argsort(predicted_ratings))[::-1]
        output = books.iloc[sorted_bok[0:list_len]][['bookTitle', 'bookAuthor']]
        out2 = predicted_ratings.sort_values(ascending=False)[0:list_len]
        print(output)
    # display results
    rec_header = list(output.columns)
    rec_header.insert(0, 'predict')

    layout = go.Layout(
        margin=dict(r=1, l=1, b=20, t=20))

    fig = go.Figure(data=[go.Table(
        columnwidth=[100, 300, 300],
        header=dict(values=rec_header,
                    line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
                    fill_color=['rgb(14, 17, 23)', 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                    align='center', font=dict(color='white', size=20), height=50
                    ),
        cells=dict(values=[np.round(out2, 2), output.bookTitle, output.bookAuthor],
                   line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
                   fill_color=[np.array(color_descends(rec)), 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                   align='center', font=dict(color='white', size=14), height=30
                   ))
    ], layout=layout)

    # get book info / covers
    url = books.iloc[sorted_bok[0:3]][['imageUrlL']].values
    info = books.iloc[sorted_bok[0:3]][['bookTitle']].values
    st.write('Deine Top auswahl')

    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)

    r1 = urllib.request.urlopen(url[0][0])
    r2 = urllib.request.urlopen(url[1][0])
    r3 = urllib.request.urlopen(url[2][0])

    col1.header(info[0][0])
    col4.image(r1.read())
    col2.header(info[1][0])
    col5.image(r2.read())
    col3.header(info[2][0])
    col6.image(r3.read())

    st.write('Alle Empfehlungen für dich:')
    st.write(fig)


def task4():
    st.write('K nearest neighbor - performance measures')
    df_ratings, ratings, books, users = get_book_data(200)

# test for performance measures





def moviePrediction_item_item_cf():
    rating = pd.read_csv('ratings.csv')
    df_rating = rating.pivot(index="movieId", columns="userId", values="rating")
    movies = pd.read_csv('movies.csv')
    df_rating_raw = df_rating.copy()
    corr_matrix = create_corr_matrix(df_rating_raw)
    predicted_ratings = item_item_cf(df_rating, corr_matrix, 1, 15) ##for movies replace 79186 with i e [1:610]
    print(get_items_item_item_cf(movies, item_item_cf( df_rating, corr_matrix,1, 10),20, movies=True)) # movies

def bookprediction_item_item_cf():
    df_rating, ratings, books, users = get_book_data(200)
    df_rating_raw = df_rating.copy()
    corr_matrix = create_corr_matrix(df_rating_raw)
    predicted_ratings = item_item_cf(df_rating, corr_matrix, 79186 , 15)
    print(get_items_item_item_cf(books, item_item_cf( df_rating, corr_matrix,79186, 10),20, movies=False))# books
#bookprediction_item_item_cf()

def all_performances(movie= True, filter_tr = 20):
    if movie is True:
        rating = pd.read_csv('ratings.csv')
        result_item = performance_item_item_cf(rating.copy())
        result_distance = performance_user_user_cf_distances(rating.copy())
        print()
    else:
        rating = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
        rating.columns = ["userId", "ISBN", "rating"]

        u = rating.userId.value_counts()
        b = rating.ISBN.value_counts()
        rating = rating[rating.userId.isin(u.index[u.gt(filter_tr)])]
        rating = rating[rating.ISBN.isin(b.index[b.gt(filter_tr)])]
        result_item = performance_item_item_cf(rating.copy(),movie = False)
        result_distance = performance_user_user_cf_distances(rating.copy(), movie = False)
        print()

#print(all_performances(False))

if __name__ == "__main__":
    main()