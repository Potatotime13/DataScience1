import pandas
import numpy as np
import plotly.graph_objects as go
from plotly.colors import n_colors
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

user_number = 10
k_users = 100
list_len = 10
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

for k in range(1, k_users + 1):
    mov = user_corr[user_number][sorted_index[k]] * df_rating[sorted_index[k]].values
    recommended_amount_of_dedotated_wam += mov / sum(user_corr.iloc[sorted_index[1:k_users + 1]][[user_number]].values)[
        0]

rec = recommended_amount_of_dedotated_wam.copy()
unseen = df_rating_raw[user_number].isnull().values
recommended_amount_of_dedotated_wam *= unseen
sorted_mov = list(np.argsort(recommended_amount_of_dedotated_wam))[::-1]
output = movies.iloc[sorted_mov[0:list_len]][['title', 'genres']]

test = user_corr.iloc[sorted_index][[user_number]]

recommended_amount_of_dedotated_wam += abs(rec.min())
recommended_amount_of_dedotated_wam *= (rec.max() + abs(rec.min())) ** -1

out2 = recommended_amount_of_dedotated_wam[sorted_mov[0:list_len]]

rec_header = list(output.columns)
rec_header.insert(0, 'predict')
colors = []
for percentage in out2:
    colors.append('rgba(255,185,15,' + str(percentage ** 3) + ')')

layout = go.Layout(
        width=1000,
        height=1000,
        margin=dict(r=20, l=10, b=10, t=10))

fig = go.Figure(data=[go.Table(
    header=dict(values=rec_header,
                fill_color=['black', 'black', 'black'],
                align='center', font=dict(color='white', size=16)
                ),
    cells=dict(values=[out2, output.title, output.genres],
               fill_color=[np.array(colors), 'rgb(39,64,139)', 'rgb(39,64,139)'],
               align='center', font=dict(color='white', size=12)
               ))
], layout=layout)

fig.show()
