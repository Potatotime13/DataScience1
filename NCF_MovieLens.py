import tensorflow as tf
import pandas as pd
import numpy as np
from lenskit.algorithms.tf import BiasedMF as bmf


class NeuMF(tf.keras.Model):
    def __init__(self, dims):
        super(NeuMF, self).__init__(name='')

        self.optimizer = tf.optimizers.Adam(0.02, 0.8, 0.95)

        # input layers
        self.con_mlp = tf.keras.layers.Concatenate(axis=1)
        self.mul_gmf = tf.keras.layers.Multiply()

        # MLP path
        self.act_mlp = tf.keras.layers.ReLU()
        self.dens_mlp1 = tf.keras.layers.Dense(dims[0], activation=tf.nn.relu)
        self.dens_mlp2 = tf.keras.layers.Dense(dims[1], activation=tf.nn.relu)
        self.dens_mlp2 = tf.keras.layers.Dense(dims[2], activation=tf.nn.relu)

        # GMF Path
        self.dens_gmf1 = tf.keras.layers.Dense(dims[3], activation=tf.nn.tanh)

        # output path
        self.con_out = tf.keras.layers.Concatenate(axis=1)
        self.dens_out = tf.keras.layers.Dense(dims[4])

    def call(self, inputs, training=False):
        input_user = inputs[0]
        input_item = inputs[1]
        x1 = self.con_mlp([input_user, input_item])
        x2 = self.mul_gmf([input_user, input_item])
        x2 = self.dens_gmf1(x2, training=training)

        x1 = self.act_mlp(x1)
        x1 = self.dens_mlp1(x1, training=training)
        x1 = self.dens_mlp2(x1, training=training)

        x = self.con_out([x1, x2])
        x = self.dens_out(x, training=training)
        return x

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


def create_valid(dataset, test_len=5000):
    ind_exc = np.random.permutation(len(dataset))[0:test_len]
    test_set = dataset.iloc[ind_exc]
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
    return dataset, test_set


def main():
    ratings = pd.read_csv('ratings.csv')

    ratings, ratings_v = create_valid(ratings)
    ratings = ratings.stack().reset_index(name='rating')
    ratings_v = ratings_v.stack().reset_index(name='rating')

    indices = ratings[['userId', 'movieId']].values
    values = ratings['rating'].values
    indices_v = ratings_v[['userId', 'movieId']].values
    values_v = ratings_v['rating'].values

    ratings.rename(columns={"userId": "user"}, inplace="True")
    ratings.rename(columns={"movieId": "item"}, inplace="True")
    mf = bmf(20, epochs=5, batch_size=1)
    mf.fit(ratings)

    item_data = pd.DataFrame(mf.item_features_).T
    user_data = pd.DataFrame(mf.user_features_).T
    item_data.columns = mf.item_index_
    user_data.columns = mf.user_index_

    mask_m = np.in1d(indices_v[:, 1], mf.item_index_)
    indices_v = indices_v[mask_m]
    values_v = values_v[mask_m]

    user_input = user_data[indices[:, 0]].T
    item_input = item_data[indices[:, 1]].T
    user_input_v = user_data[indices_v[:, 0]].T
    item_input_v = item_data[indices_v[:, 1]].T

    neu = NeuMF([20, 20, 20, 20, 1])  # MLP x 3 , GMF x 1, Out x 1
    neu.compile(loss='mse', metrics=['mse'])
    neu.fit([user_input, item_input], values, batch_size=64, epochs=20)

    neu.evaluate([user_input_v, item_input_v], values_v, batch_size=1)

    print('testlab')

    #a = neu([user_input, item_input], training=True)
    #neu.summary()
    # print(a)

    '''
    A = tf.SparseTensor(indices=indices, values=values,
                        dense_shape=[ratings.userId.max() + 1, ratings.movieId.max() + 1])
    embeddings = 20
    U = tf.Variable(tf.random.normal([A.shape[0], embeddings]), dtype=tf.float32)
    V = tf.Variable(tf.random.normal([embeddings, A.shape[1]]), dtype=tf.float32)
    df_rating = ratings.pivot(index="movieId", columns="userId", values="rating")
    del_list = []
    for ind in range(len(indices_v)):
        if indices_v[ind][0] not in mf.user_index_ or indices_v[ind][1] not in mf.user_index_:
            print(indices_v[ind])
            del_list.append(ind)
    indices_v = np.delete(indices_v, del_list, axis=0)

    optimizer = tf.optimizers.Adam(0.01, 0.8, 0.95)

    trainable_weights = [U, V]

    for step in range(2000):
        with tf.GradientTape() as tape:
            A_prime = tf.matmul(U, V)
            # indexing the result based on the indices of A that contain a value
            A_prime_sparse = tf.gather(
                tf.reshape(A_prime, [-1]),
                indices[:, 0] * tf.shape(A_prime)[1] + indices[:, 1],
            )
            loss = tf.reduce_sum(tf.metrics.mean_squared_error(A_prime_sparse, A.values))
        grads = tape.gradient(loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        if step % 20 == 0:
            print(f"Training loss at step {step}: {loss:.4f}")
            if loss < 0.8:
                break                
    user_input_v = tf.gather(U, indices_v[:, 0], axis=0).numpy()
    item_input_v = tf.gather(V, indices_v[:, 1], axis=1).numpy().T

    '''
if __name__ == "__main__":
    main()