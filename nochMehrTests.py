import pandas
import numpy as np
import plotly.graph_objects as go

ratings = pandas.read_csv('ratings.csv')
df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")

# centering
df_rating = df_rating - df_rating.mean()
df_rating = df_rating.fillna(0)
user_std = (df_rating * df_rating).mean() ** 0.5
# calc cov matrix
user_corr = df_rating.cov() / (user_std.values.reshape((-1, 1)) @ user_std.values.reshape((1, -1)))
user_corr = user_corr.fillna(0)

sorted_index = pandas.DataFrame(np.argsort(user_corr.values))
a = sorted_index.iloc[:, [608, 609]]
b = user_corr.iloc[a.iloc[:, 0].values, a.iloc[:, 1].values]
largest = []
for i in range(a.shape[0]):
    largest.append(user_corr.iloc[a.iloc[i, 0], a.iloc[i, 1]])
top = a.iloc[(list(np.argsort(largest))[::-1])[0:20]].index
largest_S = np.sort(largest)[::-1]
classes = [[] for i_1 in range(20)]
for i_2 in range(user_corr.shape[0]):
    sims = user_corr.iloc[[i_2], top].values
    j = list(np.argsort(sims))[::-1]
    classes[j[0][0]].append(int(i_2))
final_index = []
for cl in classes:
    final_index += cl
plot_surf = user_corr.iloc[final_index, final_index]
print(sorted_index)
