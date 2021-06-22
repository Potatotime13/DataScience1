import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import copy
from tmdbv3api import TMDb
from tmdbv3api import Movie
from scipy.spatial.distance import hamming, euclidean, chebyshev, cityblock
import urllib.request
import tensorflow as tf


def main():
    st.title('Data Science: Recommender Systems')
    with st.sidebar:
        st.write('Page selection')

    c_task = st.sidebar.selectbox(
        "",
        ("MovieLens", "MovieLens (item/item)", "MovieLens (AI)", "MovieLens Performance", "Books", "Books (item/item)", "Books Performance")
    )
    if c_task == "MovieLens":
        task1()
    if c_task == "MovieLens Performance":
        task2()
    if c_task == "Books":
        task3()
    if c_task == "Books Performance":
        task4()
    if c_task == "MovieLens (item/item)":
        task5()
    if c_task == "Books (item/item)":
        task6()
    if c_task == "MovieLens (AI)":
        task7()


# Data set methods
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
        percentiles += (percentiles == 0) * np.mean(percentiles[percentiles != 0])
        df_zeros = df_ratings.values == 0
        df_ratings = df_ratings + df_zeros * percentiles
        ratings["rating"] = df_ratings.stack().values

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


# recommendation methods
def pearson(df_rating, user_number, k_users, df_rating_raw, normalization):
    user_std = (df_rating * df_rating).mean() ** 0.5

    # calc correlation matrix
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
        df_rating = (df_rating - df_rating.mean()) / df_rating.var() ** 0.5
        df_rating = df_rating.fillna(0)
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


def load_results_movie():
    result_user = []
    result_ncf = []
    result_user_eu = []
    result_item = []
    for i in range(4):
        result_user.append(pd.read_csv('perf/person_user_' + str(i) + '.csv'))
        result_ncf.append(pd.read_csv('perf/ncf_mov_' + str(i) + '.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_basic.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_errors_total.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_group_actual.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_group_predicted.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_basic.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_errors_total.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_group_actual.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_group_predicted.csv'))
    return result_user, result_user_eu, result_ncf, result_item


def load_results_book():
    result_user = []
    result_user_eu = []
    result_item = []
    for i in range(4):
        result_user.append(pd.read_csv('perf/person_user_book_' + str(i) + '.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_basic.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_errors_total.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_group_actual.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_group_predicted.csv'))
    result_item.append(pd.read_csv('perf/0item_book_basic.csv'))
    result_item.append(pd.read_csv('perf/0item_book_errors_total.csv'))
    result_item.append(pd.read_csv('perf/0item_book_group_actual.csv'))
    result_item.append(pd.read_csv('perf/0item_book_group_predicted.csv'))
    return result_user, result_user_eu, result_item


def moviePrediction_item_item_cf():
    rating = pd.read_csv('ratings.csv')
    df_rating = rating.pivot(index="movieId", columns="userId", values="rating")
    movies = pd.read_csv('movies.csv')
    df_rating_raw = df_rating.copy()
    corr_matrix = create_corr_matrix(df_rating_raw)
    predicted_ratings = item_item_cf(df_rating, corr_matrix, 1, 15)  ##for movies replace 79186 with i e [1:610]
    print(get_items_item_item_cf(movies, item_item_cf(df_rating, corr_matrix, 1, 10), 20, movies=True))  # movies


def bookprediction_item_item_cf():
    df_rating, ratings, df_rating_nonzero, books, users = get_book_data(200)
    df_rating_raw = df_rating.copy()
    corr_matrix = create_corr_matrix(df_rating_raw)
    predicted_ratings = item_item_cf(df_rating, corr_matrix, 79186, 15)
    print(get_items_item_item_cf(books, item_item_cf(df_rating, corr_matrix, 79186, 10), 20, movies=False))  # books
#### end of example

def get_items_item_item_cf(item_list, predicted_ratings, list_len, movies=True, na_filler=0):
    predicted_ratings.fillna(na_filler, inplace=True)
    sorted = np.argsort(predicted_ratings)  # [::-1]
    if movies is True:
        output = item_list.iloc[sorted[0:list_len]][['title', 'genres']]
    else:
        output = item_list.iloc[sorted[0:list_len]][['bookTitle', 'bookAuthor']]
    return output


# UI helper methods
def get_favorite_movies(ratings, user_number):
    # get movies seen by the user
    df_seen = ratings.loc[:, user_number].replace(0, np.nan)
    df_seen = df_seen.dropna(how="all", axis=0)
    # prints a sorted list of the users movies
    # TODO ausgabe wird später als streamlit list erfolgen
    # print("already seen:",df_seen.sort_values(ascending=True))
    return df_seen


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


# streamlit pages
def task1():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')

    # get settings from sidebar
    user_number = st.sidebar.selectbox("User ID", (10, 12, 69, 52, 153))
    k_users = st.sidebar.selectbox("K nearest", (5, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))
    normalization = st.sidebar.selectbox("Normalization",
                                         ('centering + division by variance', 'centering', "None"))
    distance_measure = st.sidebar.selectbox("Distance measure",
                                            ('pearson', "euclidean", "manhattan (city block)",
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
    if distance_measure == "pearson":
        recommended = pearson(df_rating, user_number, k_users, df_rating_raw, normalization)
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

    st.write('your top recommendations - calculated with user/user knn')

    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)

    col1.header(info[0])
    col4.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[0])
    col2.header(info[1])
    col5.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[1])
    col3.header(info[2])
    col6.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[2])

    st.write('All recommendations for you:')
    st.write(fig)


def task2():
    # load data
    result_user, result_user_eu, result_ncf, result_item = load_results_movie()

    # header
    st.write('K nearest neighbor - performance measures')

    # get settings from sidebar
    info_shown = st.sidebar.selectbox("Measures", ("basic measures", "distribution measures"))

    # result_item, result_distance = all_performances()
    if info_shown == "distribution measures":

        categories = list(result_item[2].columns[1:])
        fig1 = go.Figure(data=[
            go.Bar(name='ncf', x=categories, y=list(result_ncf[2].iloc[4][categories])),
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[2].iloc[4][categories])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[2].iloc[4][categories])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[2].iloc[4][categories]))
        ])
        fig2 = go.Figure(data=[
            go.Bar(name='ncf', x=categories, y=list(result_ncf[2].iloc[3][categories])),
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[2].iloc[3][categories])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[2].iloc[3][categories])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[2].iloc[3][categories]))
        ])
        fig3 = go.Figure(data=[
            go.Bar(name='ncf', x=categories, y=list(result_ncf[2].iloc[1][categories])),
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[2].iloc[1][categories])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[2].iloc[1][categories])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[2].iloc[1][categories]))
        ])

        # Change display settings
        fig1.update_layout(barmode='group',
                           title=go.layout.Title(
                               text=result_item[2].iloc[4][0],
                               xref="paper",
                               x=0
                           ),
                           )
        fig2.update_layout(barmode='group',
                           title=go.layout.Title(
                               text=result_item[2].iloc[3][0],
                               xref="paper",
                               x=0
                           ),
                           )
        fig3.update_layout(barmode='group',
                           title=go.layout.Title(
                               text=result_item[2].iloc[1][0],
                               xref="paper",
                               x=0
                           ),
                           )
        # display results
        st.write(fig1)
        st.write(fig2)
        st.write(fig3)

    elif info_shown == "basic measures":

        categories = list(result_item[1].iloc[:, 0])
        fig1 = go.Figure(data=[
            go.Bar(name='ncf', x=categories, y=list(result_ncf[1].iloc[:, 1])),
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[1].iloc[:, 1])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[1].iloc[:, 1])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[1].iloc[:, 1]))
        ])
        categories2 = list(result_item[0].iloc[:, 0])
        fig2 = go.Figure(data=[
            go.Bar(name='ncf', x=categories2, y=list(result_ncf[0].iloc[:, 1])),
            go.Bar(name='item / item pearson', x=categories2, y=list(result_item[0].iloc[:, 1])),
            go.Bar(name='user / user euclidean', x=categories2, y=list(result_user_eu[0].iloc[:, 1])),
            go.Bar(name='user / user pearson', x=categories2, y=list(result_user[0].iloc[:, 1]))
        ])

        # Change display settings
        fig1.update_layout(barmode='group',
                           title=go.layout.Title(
                               text='basic measures of predictions',
                               xref="paper",
                               x=0
                           ),
                           )
        fig2.update_layout(barmode='group',
                           title=go.layout.Title(
                               text='prediction summary',
                               xref="paper",
                               x=0
                           ),
                           )
        # display results
        st.write(fig1)
        st.write(fig2)


def task3():
    #load data
    df_rating, ratings, df_rating_nonzero, books, users = get_book_data(200)

    # get settings from sidebar
    user_number = st.sidebar.selectbox("User ID", (79186, 207782))
    k_users = st.sidebar.selectbox("K nearest", (5, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))
    normalization = st.sidebar.selectbox("Normalization",
                                         ('centering + division by variance', 'centering', "None"))
    distance_measure = st.sidebar.selectbox("Distance measure",
                                            ('pearson', "euclidean", "manhattan (city block)",
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

    if distance_measure == "pearson":
        recommended = pearson(df_rating, user_number, k_users, df_rating_raw, normalization)
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
    st.write('your top recommendations - calculated with user/user knn')

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

    st.write('All recommendations for you:')
    st.write(fig)


def task4():
    # load data
    result_user, result_user_eu, result_item = load_results_book()

    # header
    st.write('K nearest neighbor - performance measures')

    # get settings from sidebar
    info_shown = st.sidebar.selectbox("Measures", ("distribution measures","basic measures", ))

    # result_item, result_distance = all_performances()
    if info_shown == "distribution measures":

        categories = [str(a) for a in np.arange(1., 11., 1)]
        fig1 = go.Figure(data=[
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[2].iloc[4][categories])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[2].iloc[4][categories])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[2].iloc[4][categories]))
        ])
        fig2 = go.Figure(data=[
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[2].iloc[3][categories])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[2].iloc[3][categories])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[2].iloc[3][categories]))
        ])
        fig3 = go.Figure(data=[
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[2].iloc[1][categories])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[2].iloc[1][categories])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[2].iloc[1][categories]))
        ])

        # Change display settings
        fig1.update_layout(barmode='group',
                           title=go.layout.Title(
                               text=result_item[2].iloc[4][0],
                               xref="paper",
                               x=0
                           ),
                           )
        fig2.update_layout(barmode='group',
                           title=go.layout.Title(
                               text=result_item[2].iloc[3][0],
                               xref="paper",
                               x=0
                           ),
                           )
        fig3.update_layout(barmode='group',
                           title=go.layout.Title(
                               text=result_item[2].iloc[1][0],
                               xref="paper",
                               x=0
                           ),
                           )

        # display results
        st.write(fig1)
        st.write(fig2)
        st.write(fig3)

    elif info_shown == "basic measures":

        categories = list(result_item[1].iloc[:, 0])
        fig1 = go.Figure(data=[
            go.Bar(name='item / item pearson', x=categories, y=list(result_item[1].iloc[:, 1])),
            go.Bar(name='user / user euclidean', x=categories, y=list(result_user_eu[1].iloc[:, 1])),
            go.Bar(name='user / user pearson', x=categories, y=list(result_user[1].iloc[:, 1]))
        ])
        categories2 = list(result_item[0].iloc[:, 0])
        fig2 = go.Figure(data=[
            go.Bar(name='item / item pearson', x=categories2, y=list(result_item[0].iloc[:, 1])),
            go.Bar(name='user / user euclidean', x=categories2, y=list(result_user_eu[0].iloc[:, 1])),
            go.Bar(name='user / user pearson', x=categories2, y=list(result_user[0].iloc[:, 1]))
        ])

        # Change display settings
        fig1.update_layout(barmode='group',
                           title=go.layout.Title(
                               text='basic measures of predictions',
                               xref="paper",
                               x=0
                           ),
                           )
        fig2.update_layout(barmode='group',
                           title=go.layout.Title(
                               text='prediction summary',
                               xref="paper",
                               x=0
                           ),
                           )
        # display results
        st.write(fig1)
        st.write(fig2)


def task5():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')

    # get settings from sidebar
    user_number = st.sidebar.selectbox("User ID", (10, 12, 69, 52, 153))
    k_users = st.sidebar.selectbox("K nearest", (5, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))

    # split the genres per movie
    movies["genres"] = movies["genres"].str.split('|')

    # rating table
    df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
    df_rating_raw = df_rating

    k_items = k_users
    corr_matrix = create_corr_matrix(df_rating_raw)
    predicted_ratings = item_item_cf(df_rating_raw, corr_matrix, user_number, k_items)  ##for movies replace 79186 with i e [1:610]
    predicted_ratings.fillna(0, inplace=True)
    recommended = predicted_ratings.copy()
    rec = recommended.copy()
    sorted_mov = list(np.argsort(predicted_ratings))[::-1]
    output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]
    out2 = predicted_ratings.sort_values(ascending=False)[0:list_len]

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

    st.write('your top recommendations - calculated with item/item knn')

    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)

    col1.header(info[0])
    col4.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[0])
    col2.header(info[1])
    col5.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[1])
    col3.header(info[2])
    col6.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[2])

    st.write('All recommendations for you:')
    st.write(fig)


def task6():
    #load data
    df_rating, ratings, df_rating_nonzero, books, users = get_book_data(200)

    # get settings from sidebar
    user_number = st.sidebar.selectbox("User ID", (79186, 207782))
    k_users = st.sidebar.selectbox("K nearest", (5, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))

    df_rating_raw = df_rating

    k_items = k_users
    corr_matrix = create_corr_matrix(df_rating_raw)
    predicted_ratings = item_item_cf(df_rating_raw, corr_matrix, user_number, k_items)  ##for movies replace 79186 with i e [1:610]
    predicted_ratings.fillna(0, inplace=True)
    recommended = predicted_ratings.copy()
    rec = recommended.copy()
    sorted_bok = list(np.argsort(predicted_ratings))[::-1]
    output = books.iloc[sorted_bok[0:list_len]][['bookTitle', 'bookAuthor']]
    out2 = predicted_ratings.sort_values(ascending=False)[0:list_len]

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
    st.write('your top recommendations - calculated with item/item knn')

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

    st.write('All recommendations for you:')
    st.write(fig)


def task7():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')

    # get settings from sidebar
    user_number = st.sidebar.selectbox("User ID", (10, 12, 69, 52, 153))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))

    # split the genres per movie
    movies["genres"] = movies["genres"].str.split('|')

    # load net
    neu = tf.keras.models.load_model('./NCF_recources', compile=True)

    # load inputs
    mfg_i = pd.read_csv('NCF_recources/mfg_i.csv')
    mfg_ib = pd.read_csv('NCF_recources/mfg_ib.csv')
    mfg_u = pd.read_csv('NCF_recources/mfg_u.csv')
    mfg_ub = pd.read_csv('NCF_recources/mfg_ub.csv')

    mlg_i = pd.read_csv('NCF_recources/mlg_i.csv')
    mlg_ib = pd.read_csv('NCF_recources/mlg_ib.csv')
    mlg_u = pd.read_csv('NCF_recources/mlg_u.csv')
    mlg_ub = pd.read_csv('NCF_recources/mlg_ub.csv')

    u_ind = pd.read_csv('NCF_recources/u_ind.csv')
    i_ind = pd.read_csv('NCF_recources/i_ind.csv')

    user_input_vg = np.ones((9512, 20)) * np.array([mfg_u.iloc[user_number, 1:]])
    item_input_vg = mfg_i.values[:, 1:]
    user_bias_vg = np.ones((9512, 1)) * mfg_ub['u_off'].iloc[10]
    item_bias_vg = mfg_ib['i_off'].values
    user_input_vm = np.ones((9512, 31)) * np.concatenate([mlg_u.values[user_number, 1:], np.array([mlg_ub['u_off'].values[user_number]])])
    item_input_vm = np.concatenate([mlg_i.values[:, 1:], np.array([mlg_ib['i_off'].values]).T], axis=1)

    y_pred = neu.predict([user_input_vg, item_input_vg, user_bias_vg, item_bias_vg, user_input_vm, item_input_vm],
                         batch_size=1)
    # normalization procedure
    rec = y_pred.T[0]
    sorted_mov = list(np.argsort(y_pred.T[0]))[::-1]
    mov_ids = i_ind['item'].iloc[sorted_mov[0:list_len]].values
    movies.set_index(movies['movieId'], inplace=True)
    output = movies.loc[mov_ids][['title', 'genres']]
    out2 = np.round(y_pred[sorted_mov[0:list_len]].T[0], 2).copy()

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
        cells=dict(values=[np.array(list(out2)), output.title, output.genres],
                   line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
                   fill_color=[np.array(color_descends(rec)), 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                   align='center', font=dict(color='white', size=14), height=30
                   ))
    ], layout=layout)

    # get movie info / covers
    links.set_index(links['movieId'], inplace=True)
    url, info = movie_url(links.loc[mov_ids[0:3]][['tmdbId']].values)

    st.write('your top recommendations - calculated with user/user knn')

    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)

    col1.header(info[0])
    col4.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[0])
    col2.header(info[1])
    col5.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[1])
    col3.header(info[2])
    col6.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[2])

    st.write('All recommendations for you:')
    st.write(fig)


if __name__ == "__main__":
    main()
