import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd


def main():
    st.title('KNN')
    with st.sidebar:
        st.write('Dataset selection')

    #task1()

    c_task = st.sidebar.selectbox(
        "",
        ("MovieLens", "Books")
    )
    if c_task == "1.1 Clustering":
        task1()


def task1():
    # read movie lens
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    st.write('Task 1')
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

            user_number = 10
            k_users = 15
            sorted_index = list(np.argsort(user_corr[user_number]))[::-1]
            recommended_amount_of_dedotated_wam = np.zeros(len(df_rating))
            for k in range(1, k_users+1):
                mov = user_corr[user_number][sorted_index[k]] * df_rating[sorted_index[k]].values
                recommended_amount_of_dedotated_wam += mov
            recommended_amount_of_dedotated_wam *= df_rating_raw[user_number].isnull().values
            sorted_mov = list(np.argsort(recommended_amount_of_dedotated_wam))[::-1]
            print(sorted_mov[0:10])


if __name__ == "__main__":
    main()
