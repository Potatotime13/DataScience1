import numpy as np
import pandas as pd
import time

start = time.time()
print("start")

def item_item_cf():
    
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    tags = pd.read_csv('tags.csv')
    links = pd.read_csv('links.csv')
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
#    corr_movies = df_rating.corr()
#    corr_movies_to_csv("corr_movies")
    user_movies = df_rating_raw[user_number]
    movie_indices = list(user_movies.index)
    counter = 0
    predictions = []
    print("loop starts!")
    for x in user_movies:
        if counter%1000 == 0: print(counter)
        if x == x:
            # find the index of the k most similar movies within the correlation matrix 
            current_movie_id = str(movie_indices[counter])
            k_most_similar = np.argpartition(corr_movies[current_movie_id],-k_items)
            k_most_similar = k_most_similar[-k_items:]
            
            # predict based on the average of the user for the k movies
            predictions.append(np.mean(df_rating_average[user_number].iloc[k_most_similar]))
            counter += 1
        
        # nan is not equal to itself, so if movie is not seen
        # the entry is nan            
        else:
            predictions.append(-50)
            counter += 1
    print("loop done!")
    print(predictions)
item_item_cf()


end = time.time()
print("time to run (mins):",(end - start)/60)
