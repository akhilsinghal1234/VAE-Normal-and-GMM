import numpy as np

# for i in range(100):
#     ch.append(np.random.choice([1,2],p=[0.6,0.4]))


import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from load_img import *

# import parameters
from params import *

# tensorflow uses channels_last
# theano uses channels_first
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)


# encoder architecture
x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(3,3),
                padding='same', activation='relu')(x)

conv_1_pool = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(conv_1)

conv_2 = Conv2D(filters1,
                kernel_size=(3, 3),
                padding='same', activation='relu')(conv_1_pool)
# ,strides=(2, 2)

conv_2_pool = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(conv_2)

conv_3 = Conv2D(filters1,
                kernel_size=(3,3),
                padding='same', activation='relu',
                strides=1)(conv_2_pool)

conv_3_pool = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(conv_3)

conv_4 = Conv2D(filters1,
                kernel_size=(3,3),
                padding='same', activation='relu',
                strides=1)(conv_3_pool)

flat = Flatten()(conv_4)
hidden1 = Dense(512, activation='relu')(flat)
h = 1024

# hidden2 = Dense(h, activation='relu')(hidden1)
hidden = Dense(intermediate_dim, activation='relu')(hidden1)
# mean and variance for latent variables
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

# sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    epsilon2 = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    
    ep = (np.random.choice([epsilon,epsilon2],p=[0.3,0.7]))                 
    return z_mean + K.exp(z_log_var) * ep

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


# decoder architecture
decoder_hid = Dense(intermediate_dim, activation='relu')
# decoder_upsample = Dense(filters * img_rows / 10 * img_cols / 2, activation='relu')
decoder_hid2 = Dense(1024, activation='relu')

decoder_upsample = Dense(64*4*4, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters1, 4, 4)
else:
    output_shape = (batch_size, 4, 4, filters1)

decoder_reshape = Reshape(output_shape[1:])

decoder_deconv_1_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3,3),
                                          strides=(4,4),
                                          padding='valid',
                                          activation='relu')

decoder_deconv_2_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(3, 3),
                                          padding='valid',
                                          activation='relu')

decoder_deconv_4_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
                                          
decoder_deconv_pad = ZeroPadding2D(padding=1)

decoder_deconv_6_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same',
                                          activation='relu') 

decoder_deconv_7 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='valid',
                                   strides=1,
                                   activation='relu')

decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='same',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
hid_decoded1 = decoder_hid2(hid_decoded)
up_decoded = decoder_upsample(hid_decoded1)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1_upsamp(reshape_decoded)
deconv_2_decoded = decoder_deconv_2_upsamp(deconv_1_decoded)

# deconv_3_decoded = decoder_deconv_3(deconv_2_decoded)
deconv_4_decoded = decoder_deconv_4_upsamp(deconv_2_decoded)
# deconv_5_decoded = decoder_deconv_5(deconv_4_decoded)
deconv_pad_decoded = decoder_deconv_pad(deconv_4_decoded)
deconv_6_decoded = decoder_deconv_6_upsamp(deconv_pad_decoded)
# deconv_7_decoded = decoder_deconv_7(deconv_6_decoded)

x_decoded_relu = decoder_deconv_7(deconv_6_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = 0.3 * (- 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        kl_loss = kl_loss + 0.7 * (- 0.5 * K.mean(z_log_var - K.square(z_mean) - K.exp(z_log_var) -2*z_mean, axis=-1))
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()
# load dataset
all_images = finger_data()                      # not normalized till now
y = [0 for i in range(len(all_images))] 
x_train, x_test, _, _ = train_test_split(all_images, y, test_size=0.1, random_state=42)
# (x_train, _), (x_test, y_test) = cifar10.load_data()
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype = np.float32)
x_train = x_train / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

# training
history = vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))
# encoder from learned model
encoder = Model(x, z_mean)

# generator / decoder from learned model
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_hid_decoded1 = decoder_hid2(_hid_decoded)
_up_decoded = decoder_upsample(_hid_decoded1)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1_upsamp(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2_upsamp(_deconv_1_decoded)

# _deconv_3_decoded = decoder_deconv_3(_deconv_2_decoded)
_deconv_4_decoded = decoder_deconv_4_upsamp(_deconv_2_decoded)
# _deconv_5_decoded = decoder_deconv_5(_deconv_4_decoded)
_deconv_pad_decoded = decoder_deconv_pad(_deconv_4_decoded)
_deconv_6_decoded = decoder_deconv_6_upsamp(_deconv_pad_decoded)

_x_decoded_relu = decoder_deconv_7(_deconv_6_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# save all 3 models for future use - especially generator
vae.save('models/e_%d_vae.h5' % (epochs))
encoder.save('models/e_%d_encoder.h5' % (epochs))
generator.save('models/e_%d_generator.h5' % (epochs))

# save training history
fname = 'models/e_%d_history.pkl' % (epochs)
with open(fname, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
