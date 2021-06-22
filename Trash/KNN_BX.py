import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
books = pd.read_csv('../BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

users = pd.read_csv('../BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ["userId", "location", "age"]

ratings = pd.read_csv('../BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ["userId", "ISBN", "rating"]

#print(ratings.loc["Book-Rating"])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

ratings = ratings.drop_duplicates(subset=["userId", "ISBN"])
duplicateRowsDF = ratings[ratings.duplicated(["userId", "ISBN"])]
print(duplicateRowsDF) ## no duplicate ratings
print(ratings.shape)

u = ratings.userId.value_counts()
b = ratings.ISBN.value_counts()

ratings = ratings[ratings.userId.isin(u.index[u.gt(5)])]
ratings = ratings[ratings.ISBN.isin(b.index[b.gt(5)])]
print(ratings.shape)
plt.hist(ratings.rating)
plt.show()


df_ratings = ratings.pivot(index="ISBN", columns="userId", values="rating")
#df_ratings = df_ratings.
print(df_ratings.replace(0, np.nan).quantile(0.75,axis=1).mean())
# file:///C:/Users/Somme/AppData/Local/Temp/Is_seeing_believing_how_recommender_system_interfa.pdf# könnte interessant sein
