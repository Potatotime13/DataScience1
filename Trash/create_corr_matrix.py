import numpy as np
import pandas as pd
import time

start = time.time()
print("start")

def item_item_cf():
    
    movies = pd.read_csv('../movies.csv')
    ratings = pd.read_csv('../ratings.csv')
    tags = pd.read_csv('../tags.csv')
    links = pd.read_csv('../links.csv')
    corr_movies = pd.read_csv("corr_movies.csv")

    user_number = 15
    k_items =  15
    list_len = 30
    normalization = "centering"

    movies["genres"] = movies["genres"].str.split('|')
    # rating table
    df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
    df_rating_raw = df_rating
    df_rating_average = df_rating.fillna(df_rating.mean(axis=1))
    df_rating = df_rating.fillna(df_rating.mean(axis=1))
    df_rating = df_rating.transpose()
    corr_movies = df_rating.corr()
    corr_movies = corr_movies.round(5)
    corr_movies.to_csv("corr_movies")

item_item_cf()


end = time.time()
print("time to run (mins):",(end - start)/60)
