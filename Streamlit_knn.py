import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def predicted_ratings_distances(df_rating, similarities, user_number, k_users, df_rating_raw, weigthing=False):
    sorted_index = list(np.argsort(similarities))[::-1][1:k_users + 1]
    # w_sum_k = rating_k * weighting vector (abhängig von sim!)
    mv_rated = df_rating.iloc[:, sorted_index]
    mv_rated = mv_rated[df_rating_raw[user_number].isnull()]
    predicted_ratings = mv_rated.mean(axis=1)
    predicted_ratings.fillna(0, inplace=True)
    return predicted_ratings


def create_valid(dataset, test_len=5000):
    ind_exc = np.random.permutation(len(dataset))[0:test_len]
    test_set = dataset.iloc[ind_exc]
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
    return dataset, test_set


def test():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')

    user_number = st.sidebar.selectbox("User ID", (10, 12, 69, 52, 153))
    k_users = st.sidebar.selectbox("K nearest", (23, 15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))
    normalization = st.sidebar.selectbox("Normalization",
                                         ('centering + division by variance', 'centering', "None"))
    distance_measure = st.sidebar.selectbox("distance_measure",
                                            ("euclidean", 'cosine', "euclidean", "manhattan (city block)", "hamming",
                                             "chebyshev"))
    train, test = create_valid(ratings)

    df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
    df_rating_raw = df_rating
    df_rating_mean = df_rating_raw.fillna(df_rating_raw.mean())
    if normalization == 'centering + division by variance':
        df_rating = (df_rating - df_rating.mean()) / df_rating.var() ** 0.5
        df_rating = df_rating.fillna(0)
    elif normalization == 'centering':
        df_rating = df_rating - df_rating.mean()
        df_rating = df_rating.fillna(0)
    elif normalization == "None":
        df_rating = df_rating.fillna(df_rating.mean())
        pass

    c = 0
    predicted = []
    actuals = []
    for x in test:
        # check if any value for a user is in test set
        if c % 100 ==0:
            print(c)
        if test[x].notna().values.any():
            similarities = similarity_calculation_distances(df_rating, distance_measure, x)
            predicted_ratings = predicted_ratings_distances(df_rating_mean, similarities, user_number, k_users,
                                                            df_rating_raw)
            index = list(test[x].dropna().index)
            actual = test[x].dropna()
            predicted.append(predicted_ratings[predicted_ratings.index.isin(index)])
            actuals.append(actual)
            c += 1
        else:
            pass
    print(predicted[5])
    print(actuals[5])
#test()

def get_favorite_movies(ratings, user_number):
    # get movies seen by the user
    df_seen = ratings.loc[:, user_number].replace(0, np.nan)
    df_seen = df_seen.dropna(how="all", axis=0)
    # prints a sorted list of the users movies
    # TODO ausgabe wird später als streamlit list erfolgen
    # print("already seen:",df_seen.sort_values(ascending=True))
    return df_seen


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
                                            ("euclidean", 'cosine', "euclidean", "manhattan (city block)", "hamming",
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
        pass

    if distance_measure == "cosine":
        recommended = cosine(df_rating, user_number, k_users, df_rating_raw, normalization)
        print("max rating:", max(recommended))
        rec = recommended.copy()
        # recommended movies

        sorted_mov = list(np.argsort(recommended))[::-1]
        output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]
        print(output)
        out2 = recommended[sorted_mov[0:list_len]]

    else:
        similarities = similarity_calculation_distances(df_rating, distance_measure, user_number)
        df_rating_mean = df_rating_raw.fillna(df_rating_raw.mean())
        predicted_ratings = predicted_ratings_distances(df_rating_mean, similarities, user_number, k_users,
                                                        df_rating_raw)
        recommended = predicted_ratings.copy()
        rec = recommended.copy()
        sorted_mov = list(np.argsort(predicted_ratings))[::-1]
        output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]
        print(predicted_ratings)
        print(output)

        out2 = predicted_ratings.sort_values(ascending=False)[0:list_len]

    # TODO (lucas) farben für alle dist maße richtig anzeigen
    #    color_grade = recommended + abs(rec.min())
    #    if rec.max() + abs(rec.min()) > 0:
    #        color_grade *= (rec.max() + abs(rec.min())) ** -1
    #    else:
    #        color_grade *= 1
    #    np.array(color_grade).sort()
    #    color_grade = np.flip(color_grade)

    # display results
    rec_header = list(output.columns)
    rec_header.insert(0, 'predict')
    #    colors = []
    #    for percentage in color_grade:
    #        colors.append('rgba(255,185,15,' + str(percentage ** 2) + ')')

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
                   fill_color=['rgb(14, 17, 23)', 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                   align='center', font=dict(color='white', size=14), height=30
                   ))
    ], layout=layout)

    #        cells=dict(values=[np.round(out2, 2), output.title, output.genres],
    #                   line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
    #                   fill_color=[np.array(colors), 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
    #                   align='center', font=dict(color='white', size=14), height=30
    #                   ))
    #    ], layout=layout)

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
    ind_exc = np.random.permutation(len(ratings))[0:5000]
    test_set = ratings.iloc[ind_exc]
    ratings.drop(index=ind_exc, inplace=True)

    with st.beta_expander("display code"):
        with st.echo('below'):
            # split the genres per movie
            movies["genres"] = movies["genres"].str.split('|')

            # rating table
            df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
            df_test_set = test_set.pivot(index="movieId", columns="userId", values="rating")
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
            errors = []
            y_pred = []
            num_err = 0

            for user_number, val in df_test_set.iteritems():
                # index of nearest users
                sorted_index = list(np.argsort(user_corr[user_number]))[::-1]

                # test set movies
                test_mov = val.dropna()
                test_mov_id = list(test_mov.index)
                train_mov_id = list(df_rating.index)
                test_mov_id = list(set(test_mov_id) & set(train_mov_id))
                test_mov = test_mov[test_mov_id]
                num_err += len(test_mov)

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
                    # errror teil nur noch drin für die life page, fällt später weg
                    err = np.sum((recommended.T - test_mov.values) ** 2)
                    errors.append(err)

    st.write("average error of a random test set containing 5000 data points:")
    error = (1 / num_err) * sum(errors)
    st.write(error)

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


def cover_url(isbn):
    pass


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
                                            ('cosine', "euclidean", "manhattan (city block)", "hamming",
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

    # display results
    rec_header = list(output.columns)
    rec_header.insert(0, 'predict')
    #    colors = []
    #    for percentage in color_grade:
    #        colors.append('rgba(255,185,15,' + str(percentage ** 2) + ')')

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
                   fill_color=['rgb(14, 17, 23)', 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                   align='center', font=dict(color='white', size=14), height=30
                   ))
    ], layout=layout)

    #        cells=dict(values=[np.round(out2, 2), output.title, output.genres],
    #                   line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
    #                   fill_color=[np.array(colors), 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
    #                   align='center', font=dict(color='white', size=14), height=30
    #                   ))
    #    ], layout=layout)

    # get movie info / covers
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


if __name__ == "__main__":
    main()
