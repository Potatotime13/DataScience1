import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main():
    st.title('KNN')
    with st.sidebar:
        st.write('Dataset selection')

    #task1()

    c_task = st.sidebar.selectbox(
        "",
        ("MovieLens", "Books")
    )
    if c_task == "MovieLens":
        task1()


def task1():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    st.write('K nearest neighbor centered cosine distance')
    user_number = st.sidebar.selectbox("User ID", (10, 12, 52, 53))
    k_users = st.sidebar.selectbox("K nearest", (15, 20))
    list_len = st.sidebar.selectbox("Recommendations", (10, 40))

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

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(output.columns)),
        cells=dict(values=[output.title, output.genres]))
    ])
    fig.update_layout(
        autosize=True,
        showlegend=False,
    )

    url = ['https://m.media-amazon.com/images/M/MV5BNGVkOTlhOTktNjZiNS00NDg3LWIxMDAtZTY5Y2E0YjllN2IxXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_UX182_CR0,0,182,268_AL_.jpg',
           'https://m.media-amazon.com/images/M/MV5BYzg0NGM2NjAtNmIxOC00MDJmLTg5ZmYtYzM0MTE4NWE2NzlhXkEyXkFqcGdeQXVyMTA4NjE0NjEy._V1_UX182_CR0,0,182,268_AL_.jpg',
           'https://m.media-amazon.com/images/M/MV5BNGVjNWI4ZGUtNzE0MS00YTJmLWE0ZDctN2ZiYTk2YmI3NTYyXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_UX182_CR0,0,182,268_AL_.jpg']
    info = ['ein Film', 'noch ein Film', 'aller guten dinge sind drei']
    #fig.show()
    st.write('Deine Top auswahl')
    st.image(url[0], caption=info[0], width=200)
    st.image(url[1], caption=info[1], width=200)
    st.image(url[2], caption=info[2], width=200)

    st.write('Alle Empfehlungen für dich:')
    st.table(output)


if __name__ == "__main__":
    main()
