import tensorflow as tf
import pandas as pd
import numpy as np
from lenskit.algorithms.tf import BiasedMF as bmf
from lenskit.algorithms.bias import Bias


class NeuMF(tf.keras.Model):
    def __init__(self, dims, in_min, in_max):
        super(NeuMF, self).__init__(name='')
        self.in_min = in_min
        self.in_max = in_max
        self.optimizer = tf.optimizers.Adam()

        # input layers
        self.con_mlp = tf.keras.layers.Concatenate(axis=1)
        self.mul_gmf = tf.keras.layers.Multiply()

        # MLP path
        self.act_mlp = tf.keras.layers.ReLU()
        self.dens_mlp1 = tf.keras.layers.Dense(dims[0], activation=tf.nn.relu)
        self.dens_mlp2 = tf.keras.layers.Dense(dims[1], activation=tf.nn.relu)
        self.dens_mlp3 = tf.keras.layers.Dense(dims[2], activation=tf.nn.relu)

        # GMF Path
        self.dens_gmf1 = tf.keras.layers.Dense(dims[3], activation=tf.nn.relu)

        # output path
        self.con_out = tf.keras.layers.Concatenate(axis=1)
        self.dens_out = tf.keras.layers.Dense(dims[4], activation=tf.nn.sigmoid)

    def call(self, inputs, training=False, mask=None):
        input_user_g = inputs[0]
        input_item_g = inputs[1]
        input_bias_u = inputs[2]
        input_bias_i = inputs[3]
        input_user_m = inputs[4]
        input_item_m = inputs[5]

        x1 = self.con_mlp([input_user_m, input_item_m])
        x2 = self.mul_gmf([input_user_g, input_item_g])
        x2 = self.con_out([x2, input_bias_u, input_bias_i])
        x2 = self.dens_gmf1(x2, training=training)

        x1 = self.act_mlp(x1)
        x1 = self.dens_mlp1(x1, training=training)
        x1 = self.dens_mlp2(x1, training=training)
        x1 = self.dens_mlp3(x1, training=training)

        x = self.con_out([x1, x2])
        x = self.dens_out(x, training=training)
        return self.in_min + x * (self.in_max - self.in_min)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs, y = data

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss_ = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def get_book_data(filter_tr, like_to_value=True):
    # load data from csv
    books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                     'imageUrlL']

    users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    users.columns = ["userId", "location", "age"]

    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings.columns = ["userId", "ISBN", "rating"]

    ratings = ratings.drop_duplicates(subset=["userId", "ISBN"])

    # filter dataset for users / items with much interaction
    u = ratings.userId.value_counts()
    b = ratings.ISBN.value_counts()

    ratings = ratings[ratings.userId.isin(u.index[u.gt(filter_tr)])]
    ratings = ratings[ratings.ISBN.isin(b.index[b.gt(filter_tr)])]

    # create table
    df_ratings = ratings.pivot(index="ISBN", columns="userId", values="rating")
    df_rating_nonzero = ratings.loc[ratings["rating"].values != 0]
    if like_to_value:
        percentiles = df_ratings.describe(include='all').iloc[6].values
        df_zeros = df_ratings.values == 0
        df_ratings = df_ratings + df_zeros * percentiles
        df_zeros = df_ratings == 0
        df_ratings = df_ratings + df_zeros * np.mean(percentiles[percentiles != 0])
        ratings["rating"] = df_ratings.stack().values

    return df_ratings, ratings, df_rating_nonzero, books, users


def create_valid(dataset, test_len=5000, movie=True):
    dataset = dataset.reset_index()
    ind_exc = np.random.permutation(len(dataset))[0:test_len]
    test_set = dataset.iloc[ind_exc]
    if movie is True:
        dataset.drop(index=ind_exc, inplace=True)
        use_v = test_set['userId'].unique()
        mov_v = test_set['movieId'].unique()
        use_t = dataset['userId'].unique()
        mov_t = dataset['movieId'].unique()
        index_u = list(np.intersect1d(use_v, use_t))
        index_m = list(np.intersect1d(mov_v, mov_t))
        dataset = dataset.pivot(index="movieId", columns="userId", values="rating")
        test_set = test_set.pivot(index="movieId", columns="userId", values="rating")
        dataset = dataset[index_u]
        test_set = test_set.loc[index_m][index_u]
    else:
        dataset.drop(index=ind_exc, inplace=True)
        use_v = test_set['userId'].unique()
        mov_v = test_set['ISBN'].unique()
        use_t = dataset['userId'].unique()
        mov_t = dataset['ISBN'].unique()
        index_u = list(np.intersect1d(use_v, use_t))
        index_m = list(np.intersect1d(mov_v, mov_t))
        dataset = dataset.pivot(index="ISBN", columns="userId", values="rating")
        test_set = test_set.pivot(index="ISBN", columns="userId", values="rating")
        dataset = dataset[index_u]
        test_set = test_set.loc[index_m][index_u]

    return dataset, test_set


def main():
    #movie = False
    movie = True
    ratings = pd.read_csv('ratings.csv')
    #df_ratings, ratings, ratings_nz, books, users = get_book_data(20)
    in_min = ratings['rating'].min()
    in_max = ratings['rating'].max()

    ratings, ratings_v = create_valid(ratings, movie=movie)
    ratings = ratings.stack().reset_index(name='rating')
    ratings_v = ratings_v.stack().reset_index(name='rating')

    if movie:
        ratings.rename(columns={"userId": "user"}, inplace="True")
        ratings.rename(columns={"movieId": "item"}, inplace="True")
        ratings_v.rename(columns={"userId": "user"}, inplace="True")
        ratings_v.rename(columns={"movieId": "item"}, inplace="True")
    else:
        ratings.rename(columns={"userId": "user"}, inplace="True")
        ratings.rename(columns={"ISBN": "item"}, inplace="True")
        ratings_v.rename(columns={"userId": "user"}, inplace="True")
        ratings_v.rename(columns={"ISBN": "item"}, inplace="True")
        ratings['item'] = np.array([i for i in range(ratings.shape[0])])

    indices = ratings[['user', 'item']].values
    values = ratings['rating'].values
    indices_v = ratings_v[['user', 'item']].values
    values_v = ratings_v['rating'].values

    bias = Bias(damping=10)
    mf_g = bmf(20, bias=bias, epochs=1, batch_size=8)
    mf_g.fit(ratings)
    mf_m = bmf(30, bias=bias, epochs=1, batch_size=8)
    mf_m.fit(ratings)

    item_data_g = pd.DataFrame(mf_g.item_features_).T
    user_data_g = pd.DataFrame(mf_g.user_features_).T
    item_data_g.columns = mf_g.item_index_
    user_data_g.columns = mf_g.user_index_

    mask_m = np.in1d(indices_v[:, 1], mf_g.item_index_)
    indices_v = indices_v[mask_m]
    values_v = values_v[mask_m]

    item_bias_g = (pd.DataFrame(mf_g.bias.item_offsets_).T[indices[:, 1]]).T
    user_bias_g = (pd.DataFrame(mf_g.bias.user_offsets_).T[indices[:, 0]]).T
    item_bias_vg = (pd.DataFrame(mf_g.bias.item_offsets_).T[indices_v[:, 1]]).T
    user_bias_vg = (pd.DataFrame(mf_g.bias.user_offsets_).T[indices_v[:, 0]]).T

    user_input_g = user_data_g[indices[:, 0]].T
    item_input_g = item_data_g[indices[:, 1]].T
    user_input_vg = user_data_g[indices_v[:, 0]].T
    item_input_vg = item_data_g[indices_v[:, 1]].T

    item_data_m = pd.DataFrame(mf_m.item_features_).T
    user_data_m = pd.DataFrame(mf_m.user_features_).T
    item_data_m.columns = mf_m.item_index_
    user_data_m.columns = mf_m.user_index_

    mask_m = np.in1d(indices_v[:, 1], mf_m.item_index_)
    indices_v = indices_v[mask_m]
    values_v = values_v[mask_m]

    item_bias_m = pd.DataFrame(mf_m.bias.item_offsets_).T[indices[:, 1]]
    user_bias_m = pd.DataFrame(mf_m.bias.user_offsets_).T[indices[:, 0]]
    item_bias_vm = pd.DataFrame(mf_m.bias.item_offsets_).T[indices_v[:, 1]]
    user_bias_vm = pd.DataFrame(mf_m.bias.user_offsets_).T[indices_v[:, 0]]

    user_input_m = pd.concat([user_data_m[indices[:, 0]], user_bias_m], axis=0).T
    item_input_m = pd.concat([item_data_m[indices[:, 1]], item_bias_m], axis=0).T
    user_input_vm = pd.concat([user_data_m[indices_v[:, 0]], user_bias_vm], axis=0).T
    item_input_vm = pd.concat([item_data_m[indices_v[:, 1]], item_bias_vm], axis=0).T

    neu = NeuMF([80, 40, 20, 20, 1], in_min, in_max)  # MLP x 3 , GMF x 1, Out x 1
    neu.compile(loss='mse', metrics=['mse'])
    with tf.device('/GPU:0'):
        hist = neu.fit([user_input_g, item_input_g, user_bias_g, item_bias_g, user_input_m, item_input_m], values, batch_size=64, epochs=20)

    neu.evaluate([user_input_vg, item_input_vg, user_bias_vg, item_bias_vg, user_input_vm, item_input_vm], values_v, batch_size=1)

    y_pred = neu.predict([user_input_vg, item_input_vg, user_bias_vg, item_bias_vg, user_input_vm, item_input_vm], batch_size=1)

    classes = pd.DataFrame(ratings.pivot("item", "user", "rating").mean()).loc[indices_v[:, 0]]

    summary = np.concatenate([y_pred.T, np.array([values_v]), classes.T])

    return summary


if __name__ == "__main__":
    for i in range(1, 5):
        sum_temp = main()
        np.save('perf/NCF_summaries/sum_'+str(i), sum_temp)
        print(i)
    print('fin')
