import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers.merge import _Merge


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

def cls_loss(y_true, y_pred):
    return cls_weight * tf.keras.losses.categorical_crossentropy(y_true, y_pred)

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, x_hat, gradient_penalty_weight):
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, x_hat)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

# Generator
def build_G():
    g_input_c = tf.keras.Input(shape=(attribute_dim, ))
    g_input_z = tf.keras.Input(shape=(noise_dim,))
    g_inputs = tf.keras.layers.concatenate([g_input_c, g_input_z])
    g1 = Dense(4096, activation='linear')(g_inputs)
    g1_lr = tf.keras.layers.LeakyReLU(alpha=0.2)(g1)
    g2 = Dense(cnn_feature_dim, activation='relu')(g1_lr)

    return Model(input = [g_input_c, g_input_z], output=g2)


# Discriminator
def build_D():
    d_input_c = tf.keras.Input(shape=(attribute_dim,))
    d_input_X = tf.keras.Input(shape=(cnn_feature_dim,))
    d_inputs = tf.keras.layers.concatenate([d_input_X, d_input_c])
    d1 = Dense(2048, activation='linear')(d_inputs)
    d1_lr = tf.keras.layers.LeakyReLU(alpha=0.2)(d1)
    d2 = Dense(1, activation='linear')(d1_lr)

    return Model(input=[d_input_X, d_input_c], output=d2)


# Train pre-trained softmax classifier for GAN
def pre_train_cls():
    pre_cls = Sequential()
    pre_cls.add(Dense(num_seen, activation='softmax'))

    pre_cls.summary()
    cls_opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    pre_cls.complie(loss='softmax_cross_entropy', optimizer=cls_opt)
    early_stopping = EarlyStopping(monitor='val_loss',
                                    min_delta=0.001,
                                    patience=1,
                                    verbose=0)
    pre_cls.fit(X_feature, Y, batch_size=128, epochs=20, callbacks=[early_stopping], shuffle=True, validation_split=0.1)
    pre_cls.trainable = False
    return pre_cls

def build_G_model(G, D, cls):
    D.trainable = False
    gm_input_z = tf.keras.Input(shape=(noise_dim,))
    gm_input_c = tf.keras.Input(shape=(attribute_dim,))
    gen_features = G([gm_input_c, gm_input_z])
    d = D([gen_features, gm_input_c])
    y = cls(gen_features)

    return Model(input=[gm_input_c, gm_input_z], output=[d, y])


def build_D_model(G, D):
    D.trainable = True
    G.trainable = False
    dm_input_z = tf.keras.Input(shape=(noise_dim,))
    dm_input_X = tf.keras.Input(shape=(cnn_feature_dim,))
    dm_input_c = tf.keras.Input(shape=(attribute_dim,))
    gen_features = G([dm_input_c, dm_input_z])
    fake = D([gen_features, dm_input_c])
    real = D([dm_input_X, dm_input_c])
    X_for_pen = RandomWeightedAverage()([dm_input_X, gen_features])
    grad_pen = gradient_penalty()

# predefine of all models
G = build_G()
D = build_D()
pre_cls = pre_train_cls()

G_model = build_G_model(G, D, cls)

D_model = build_D_model(G, D, cls)


# training GAN
reals = np.ones((batch_size, 1))
fakes = -np.ones((batch_size, 1))

for epoch in range(epochs):

    # get input batch
    X_train =
    c_train =
    y_train =

    # generate fake features
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    fake_samples = G.predict([c_train, noise])


