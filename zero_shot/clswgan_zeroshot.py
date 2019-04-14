import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers.merge import _Merge
from tensorflow.keras.layers import Layer


num_seen = 5

attribute_dim = 20
noise_dim = 128
cnn_feature_dim = 2048

lamda = 10
beta = 1

epochs = 50
batch_size = 64
D_iters = 5

cls_weight = 1

X_feature = # CNN feature inputs
Y =         # labels

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class RandomWeightedAverage(_Merge):
    """
    Merge x and x\tilde to x\hat
    """
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class GrandNorm(Layer):
    """
    Layer to calculate the norm of the gradient
    """
    def __init__(self, **kwargs):
        super(GrandNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GrandNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


# Generator
def build_G():
    """
    :return: keras Model with inputs: an Attribute Vector
                                      a Noise Vector
                              output: a CNN Feature Vector
    """
    g_input_c = tf.keras.Input(shape=(attribute_dim, ))
    g_input_z = tf.keras.Input(shape=(noise_dim,))
    g_inputs = tf.keras.layers.concatenate([g_input_c, g_input_z])
    g1 = Dense(4096, activation='linear')(g_inputs)
    g1_lr = tf.keras.layers.LeakyReLU(alpha=0.2)(g1)
    g2 = Dense(cnn_feature_dim, activation='relu')(g1_lr)

    return Model(input = [g_input_c, g_input_z], output=g2)


# Discriminator
def build_D():
    """
    :return: keras Model with inputs: CNN Feature Vector
                                      Attribute Vector
                              output: wGAN output (real value)
    """
    d_input_c = tf.keras.Input(shape=(attribute_dim,))
    d_input_X = tf.keras.Input(shape=(cnn_feature_dim,))
    d_inputs = tf.keras.layers.concatenate([d_input_X, d_input_c])
    d1 = Dense(2048, activation='linear')(d_inputs)
    d1_lr = tf.keras.layers.LeakyReLU(alpha=0.2)(d1)
    d2 = Dense(1, activation='linear')(d1_lr)

    return Model(input=[d_input_X, d_input_c], output=d2)


# Train pre-trained softmax classifier for GAN
def pre_train_cls():
    """
    :return:  a keras soft-max classifier, pre-trained with training data
    """
    pre_cls = Sequential()
    pre_cls.add(Dense(num_seen, activation='softmax'))

    pre_cls.summary()
    cls_opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    pre_cls.complie(loss='softmax_cross_entropy', optimizer=cls_opt)
    early_stopping = EarlyStopping(monitor='val_loss',
                                    min_delta=0.001,
                                    patience=1,
                                    verbose=0)
    pre_cls.fit(X_train, Y_train, batch_size=128, epochs=20, callbacks=[early_stopping], shuffle=True, validation_split=0.1)
    pre_cls.trainable = False
    return pre_cls



# predefine of all models
G = build_G()
D = build_D()
pre_cls = pre_train_cls()

# define generator model for compiling
D.trainable = False
input_z = tf.keras.Input(shape=(noise_dim,))
input_c = tf.keras.Input(shape=(attribute_dim,))
gen_features = G([input_c, input_z])
d = D([gen_features, input_c])
y = pre_cls(gen_features)

G_model = Model(input=[input_c, input_z], output=[d, y])

G_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9),
                        loss=[wasserstein_loss, 'categorical_crossentropy'],
                loss_weights=[1, beta])

# define discriminator model for compiling
D.trainable = True
G.trainable = False
input_X = tf.keras.Input(shape=(cnn_feature_dim,))
fake = D([gen_features, input_c])
real = D([input_X, input_c])
X_for_pen = RandomWeightedAverage()([input_X, gen_features])
d_for_pen = D([X_for_pen, input_c])
norm = GrandNorm()([d_for_pen, X_for_pen])

D_model = Model(input=[input_X, input_c, input_z], output=[real, fake, norm])
D_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9),
                        loss=[wasserstein_loss, wasserstein_loss, 'mse'],
                loss_weights=[1, -1, lamda])


# training GAN
reals = np.ones((batch_size, 1))
fakes = -np.ones((batch_size, 1))
grad_pens = np.ones((batch_size, 1))

for epoch in range(epochs):

    # get input batch
    X_train =
    c_train =
    y_train =
    for i in range(0, n_train, batch_size):
        for j in range(D_iters):
            # generate fake features
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            D_model.train_on_batch([X_batch, c_batch, noise], [reals, fakes, grad_pens])

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        G_model.train_on_batch([c_batch, noise], [reals, y_batch])




