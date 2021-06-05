import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tmdbv3api import TMDb
from tmdbv3api import Movie


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


def task1():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')
    st.write('K nearest neighbor centered cosine distance')
    user_number = st.sidebar.selectbox("User ID", (10, 12, 52, 53))
    k_users = st.sidebar.selectbox("K nearest", (15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))

    # split the genres per movie
    movies["genres"] = movies["genres"].str.split('|')

    # rating table
    df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
    df_rating_raw = df_rating

    # centering
    df_rating = df_rating - df_rating.mean()
    df_rating = df_rating.fillna(0)
    user_std = (df_rating * df_rating).mean() ** 0.5
    # calc cov matrix
    user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
    user_corr = user_corr.fillna(0)

    sorted_index = list(np.argsort(user_corr[user_number]))[::-1]
    recommended = np.zeros(len(df_rating))

    for k in range(1, k_users+1):
        mov = user_corr[user_number][sorted_index[k]] * df_rating[sorted_index[k]].values
        recommended += mov / sum(user_corr.iloc[sorted_index[1:k_users + 1]][[user_number]].values)[0]

    rec = recommended.copy()
    unseen = df_rating_raw[user_number].isnull().values
    recommended *= unseen
    sorted_mov = list(np.argsort(recommended))[::-1]
    output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]

    recommended += abs(rec.min())
    recommended *= (rec.max() + abs(rec.min())) ** -1

    out2 = recommended[sorted_mov[0:list_len]]

    rec_header = list(output.columns)
    rec_header.insert(0, 'predict')
    colors = []
    for percentage in out2:
        colors.append('rgba(255,185,15,' + str(percentage ** 3) + ')')

    layout = go.Layout(
        margin=dict(r=1, l=1, b=20, t=20))

    fig = go.Figure(data=[go.Table(
        header=dict(values=rec_header,
                    line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
                    fill_color=['rgb(14, 17, 23)', 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
                    align='center', font=dict(color='white', size=18)
                    ),
        cells=dict(values=[np.round(out2, 2), output.title, output.genres],
                   line_color=['rgb(49, 51, 63)', 'rgb(49, 51, 63)', 'rgb(49, 51, 63)'],
                   fill_color=[np.array(colors), 'rgb(14, 17, 23)', 'rgb(14, 17, 23)'],
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
    #st.table(output)


def task2():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')
    st.write('K nearest neighbor centered cosine distance')
    user_number = 10
    k_users = 10
    list_len = 10
    with st.beta_expander("display code"):
        with st.echo('below'):
            # rating table
            df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
            df_rating_raw = df_rating

            # centering
            df_rating = df_rating - df_rating.mean()
            df_rating = df_rating.fillna(0)
            user_std = (df_rating * df_rating).mean() ** 0.5
            # calc cov matrix
            user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
            user_corr = user_corr.fillna(0)

            sorted_index = list(np.argsort(user_corr[user_number]))[::-1]
            recommended_amount_of_dedotated_wam = np.zeros(len(df_rating))
            for k in range(1, k_users+1):
                mov = user_corr[user_number][sorted_index[k]] * df_rating[sorted_index[k]].values
                recommended_amount_of_dedotated_wam += mov
            recommended_amount_of_dedotated_wam *= df_rating_raw[user_number].isnull().values
            sorted_mov = list(np.argsort(recommended_amount_of_dedotated_wam))[::-1]
            output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]


if __name__ == "__main__":
    main()
