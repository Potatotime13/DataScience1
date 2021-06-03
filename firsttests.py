import pandas
from IPython.display import display
# read movie lens
movies = pandas.read_csv('movies.csv')
ratings = pandas.read_csv('ratings.csv')
tags = pandas.read_csv('tags.csv')

# split the genres per movie
movies["genres"] = movies["genres"].str.split('|')
genres = []
for index, movie in movies.iterrows():
    for g in movie['genres']:
        if g not in genres:
            genres.append(g)

# give the genres ids
genres.sort()
gen_dict = []
count = 0
for g in genres:
    gen_dict.append((g, count))
    count += 1
gen_dict = dict(gen_dict)

# create a datatable with compressed info
movie_map = movies['movieId'].values.tolist()
for index, row in tags.iterrows():
    i = movie_map.index(row['movieId'])
    r = row['tag']

# rating table
df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")

# centering
df_rating = df_rating-df_rating.mean()
df_rating = df_rating.fillna(0)
user_std = (df_rating * df_rating).mean()**0.5
# calc cov matrix
user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
user_corr = user_corr.fillna(0)

display(user_corr[53])



