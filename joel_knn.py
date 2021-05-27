import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine, hamming, euclidean, minkowski, cityblock

links = pd.read_csv("links.csv")
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
print(ratings.head())
# print(links.head())
# print(movies.iloc[1,:])
# print(ratings.head())

# turn userid as columns and movies as rows. Ratings are the matrix entries
df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
print(df_rating.head())
##pd.set_option('display.max_rows', None)
##pd.set_option('display.max_columns', None)
##pd.set_option('display.width', None)
##pd.set_option('display.max_colwidth', -1)
##print(df_rating.iloc[:,52])

# standardise:Problem user nummer 53
# df_rating = (df_rating - df_rating.mean(axis=0,skipna=True))/df_rating.std(axis=0,skipna=True)

# normalize between 0 and 1
df_rating = (df_rating - df_rating.min()) / (df_rating.max() - df_rating.min())
print(df_rating.mean(axis=0, skipna=True))

df_rating = df_rating.fillna(0)

similarities = []
user_number = 10
users = []
for x in range(df_rating.shape[1]):
    similarities.append(1 / (1 + euclidean(df_rating.iloc[:, user_number], df_rating.iloc[:, x])))
    users.append(x)

sorted_index = list(np.argsort(similarities))[::-1]
users = np.array(users)[sorted_index]
similarities = np.array(similarities)[sorted_index]

# dictionary which matches movie id and title
movie_dict = dict(zip(movies.loc[:, 'movieId'], movies.loc[:, 'title']))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# print(df_rating.iloc[:,52])
for x in range(6):
    df = df_rating.iloc[:, users[x]].replace(0, np.nan)
    df = df.dropna(how="all", axis=0)
    df = df.rename(index=movie_dict)
    print("mean rating:", df.mean(axis=0))
    print("name:", users[x])
    print("sim:", similarities[x])
    print()
    print(df)
    print("#####")
    print()

df = df_rating.iloc[:, user_number].replace(0, np.nan)
df = df.dropna(how="all", axis=0)
df = df.rename(index=movie_dict)
print(df.sort_values())

### calc recommendations:
k = 15
number_of_recommendations = 15
k_nearest = df_rating.iloc[:, users[0:k]]

# remove movies already seen by the user
k_nearest = k_nearest[k_nearest.iloc[:, 0] == 0]

# dont count unseen movies
k_nearest = k_nearest.replace(0, np.nan)

k_nearest_rating = k_nearest.mean(axis=1)

k_nearest_rating.rename(index=movie_dict, inplace=True)
print(k_nearest_rating.nlargest(number_of_recommendations))
print("distances:", similarities[0:k])
# standardise?