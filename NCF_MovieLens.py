import tensorflow as tf


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class NeuMF(tf.keras.Model):
    def __init__(self, dims):
        super(NeuMF, self).__init__(name='')

        # input layers
        self.con_mlp = tf.keras.layers.Concatenate(axis=1)
        self.mul_gmf = tf.keras.layers.Multiply()

        # MLP path
        self.act_mlp = tf.keras.layers.ReLU()
        self.dens_mlp1 = tf.keras.layers.Dense(dims[0], activation='relu')
        self.dens_mlp2 = tf.keras.layers.Dense(dims[1], activation='relu')
        self.dens_mlp3 = tf.keras.layers.Dense(dims[2], activation='relu')
        self.dens_mlp4 = tf.keras.layers.Dense(dims[3], activation='relu')

        # output path
        self.con_out = tf.keras.layers.Concatenate(axis=1)
        self.dens_out = tf.keras.layers.Dense(dims[4])

    def call(self, input_user, input_item, training=False):
        x1 = self.con_mlp([input_user, input_item])
        x2 = self.mul_gmf([input_user, input_item])

        x1 = self.act_mlp(x1)
        x1 = self.dens_mlp1(x1)
        x1 = self.dens_mlp2(x1)
        x1 = self.dens_mlp3(x1)
        x1 = self.dens_mlp4(x1)

        x = self.con_out([x1, x2])
        x = self.dens_out(x)
        return x


neu = NeuMF([10, 20, 30, 30, 1])
a = neu(tf.zeros([1000000, 10]), tf.ones([1000000, 10]))
neu.summary()
print(a)
