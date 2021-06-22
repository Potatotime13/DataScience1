import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tmdbv3api import TMDb
from tmdbv3api import Movie
### change
from scipy.spatial.distance import cosine, hamming, euclidean, chebyshev, cityblock


# work done:
# added no standardisation option
# addded 0 to 1 standardisation
# added fill na for no and 0-1 normalisation ( fill average)
# placed fillna(0) to the "around 0 standardisation
# implemented different distance/similarity measures, put the one for correlation among them

def task1():
    # read movie lens
    movies = pd.read_csv('../Datasets/movies.csv')
    ratings = pd.read_csv('../Datasets/ratings.csv')
    tags = pd.read_csv('../Datasets/tags.csv')
    links = pd.read_csv('../Datasets/links.csv')
##    st.write('K nearest neighbor centered cosine distance')

    # get settings from sidebar
##    user_number = st.sidebar.selectbox("User ID", (10, 12, 52, 53))
##    k_users = st.sidebar.selectbox("K nearest", (15, 20))
##    list_len = st.sidebar.selectbox("Recommendations", (10, 40))

### change
##    normalization = st.sidebar.selectbox("Normalization", ('centering', 'centering + division by variance', "0-1 normalizatoin", "None"))


########
    user_number = 15
    k_users =  15
    list_len = 30
    normalization = "0-1 normalization"
    distance_measure = "chebyshev"
########


    # split the genres per movie
    movies["genres"] = movies["genres"].str.split('|')

    # rating table
    df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
    df_rating_raw = df_rating

    # normalization procedure
    if normalization == 'centering + division by variance':
        df_rating = (df_rating - df_rating.mean()) / df_rating.var() ** 0.5
        df_rating = df_rating.fillna(0)
        user_std = (df_rating * df_rating).mean() ** 0.5
    elif normalization == 'centering':
        df_rating = df_rating - df_rating.mean()
        df_rating = df_rating.fillna(0)
        user_std = (df_rating * df_rating).mean() ** 0.5
## changes
    elif normalization == "0-1 normalization":
        df_rating =(df_rating-df_rating.min())/(df_rating.max()-df_rating.min())
        df_rating = df_rating.fillna(df_rating.average())
    ###
    elif normalization == "None":
        df_rating = df_rating.fillna(df_rating.average())
        pass

    



    ### calc sim with given distance measure
    distances = []
    similarities = []
    # sry :(
    for x in range(df_rating.shape[1]):
        if distance_measure == "cosine":
            dist = cosine(df_rating.loc[:,user_number],df_rating.iloc[:,x])
            distances.append(dist)
            similarities.append(1-dist)
        elif distance_measure == "correlation":
            # calc cov matrix
            user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
            #print(user_corr)
            user_corr = user_corr.fillna(0)
            similarities = user_corr
            break
        elif distance_measure == "euclidean":
            dist = euclidean(df_rating.loc[:,user_number],df_rating.iloc[:,x])
            distances.append(dist)
            similarities.append(1/(1+dist))
        elif distance_measure == "manhattan (city block)":
            dist = cityblock(df_rating.loc[:,user_number],df_rating.iloc[:,x])
            distances.append(dist)
            similarities.append(1/(1+dist))
        elif distance_measure == "hamming":
            dist = hamming(df_rating.loc[:,user_number],df_rating.iloc[:,x])
            distances.append(dist)
            similarities.append(1-dist)
        elif distance_measure == "chebyshev":
            dist = chebyshev(df_rating.loc[:,user_number],df_rating.iloc[:,x])
            distances.append(dist)
            similarities.append(1/(1+dist))


    sorted_index = list(np.argsort(similarities))[::-1][1:k_users+1]
    # get the k best similarities and distances
    sim_k = np.array(similarities)[sorted_index]
    dist_k = np.array(distances)[sorted_index]
    ratings_k = df_rating_raw.iloc[:, sorted_index].values
    # w_sum_k = rating_k * weighting vector (abhängig von sim!)
    mv_rated = df_rating_raw.iloc[:, sorted_index].notnull().values
    seen_sim_len = mv_rated @ sim_k
    seen_sim_len = 1 / (seen_sim_len + (seen_sim_len == 0))
    #recommended = w_sum_k * seen_sim_len
    recommended = seen_sim_len
    rec = recommended.copy()

    # recommended movies
    unseen = df_rating_raw[user_number].isnull().values
    recommended = recommended.T[0] * unseen
    sorted_mov = list(np.argsort(recommended))[::-1]
    output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]    

    print(output)
    print(similarities)


    # index of nearest users
    user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
    user_corr = user_corr.fillna(0)
    sorted_index = list(np.argsort(user_corr[user_number]))[::-1]

    # sum of their ratings weighted by the corr
    corr_k = user_corr.iloc[sorted_index[1:k_users+1]][[user_number]].values
    #print(corr_k)
    
    ratings_k = df_rating.iloc[:, sorted_index[1:k_users+1]].values
    w_sum_k = ratings_k @ corr_k
    mv_rated = df_rating_raw.iloc[:, sorted_index[1:k_users + 1]].notnull().values
    seen_sim_len = mv_rated @ corr_k
    seen_sim_len = 1 / (seen_sim_len + (seen_sim_len == 0))
    recommended = w_sum_k * seen_sim_len
    rec = recommended.copy()

    # recommended movies
    unseen = df_rating_raw[user_number].isnull().values
    recommended = recommended.T[0] * unseen
    sorted_mov = list(np.argsort(recommended))[::-1]
    output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]

    # percent of best rating
##    color_grade = recommended + abs(rec.min())
##    if rec.max() + abs(rec.min()) > 0:
##        color_grade *= (rec.max() + abs(rec.min())) ** -1
##    else:
##        color_grade *= 1
##    color_grade.sort()
##    color_grade = np.flip(color_grade)

    # display results
    out2 = recommended[sorted_mov[0:list_len]]

##rec_header = list(output.columns)
##rec_header.insert(0, 'predict')
##    colors = []
##    for percentage in color_grade:
##        colors.append('rgba(255,185,15,' + str(percentage ** 2) + ')')
##
##    layout = go.Layout(
##        margin=dict(r=1, l=1, b=20, t=20))
##
##    fig = go.Figure(data=[go.Table(
##        columnwidth=[100, 300, 300],
##        header=dict(values=rec_header,
##                    line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
##                    fill_color=['rgb(14, 17, 23)', 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
##                    align='center', font=dict(color='white', size=20), height=50
##                    ),
##        cells=dict(values=[np.round(out2, 2), output.title, output.genres],
##                   line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
##                   fill_color=[np.array(colors), 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
##                   align='center', font=dict(color='white', size=14), height=30
##                   ))
##    ], layout=layout)

    # get movie info / covers
##    url, info = movie_url(links.iloc[sorted_mov[0:3]][['tmdbId']].values)
##
##    st.write('Deine Top auswahl')
##
##    col1, col2, col3 = st.beta_columns(3)
##    col4, col5, col6 = st.beta_columns(3)
##
##    col1.header(info[0])
##    col4.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[0])
##    col2.header(info[1])
##    col5.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[1])
##    col3.header(info[2])
##    col6.image('https://www.themoviedb.org/t/p/w600_and_h900_bestv2/' + url[2])
##
##    st.write('Alle Empfehlungen für dich:')
##    st.write(fig)

task1()
def task2():
    # read movie lens
    movies = pd.read_csv('../Datasets/movies.csv')
    ratings = pd.read_csv('../Datasets/ratings.csv')
    tags = pd.read_csv('../Datasets/tags.csv')
    links = pd.read_csv('../Datasets/links.csv')
    st.write('K nearest neighbor centered cosine distance')

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
                        recommended = w_sum_k * seen_sim_len * df_rating_raw[user_number].var() ** 0.5 + df_rating_raw[user_number].mean()
                    elif normalization == 'centering':
                        recommended = w_sum_k * seen_sim_len + df_rating_raw[user_number].mean()
                    err = np.sum(np.abs(recommended.T - test_mov.values))
                    errors.append(err)

    st.write("average error of a random test set containing 5000 data points:")
    error = (1/num_err) * sum(errors)
    st.write(error)

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


