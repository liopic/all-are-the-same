
from typing import List
from config import TMP_DIR
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Reshape, Conv2DTranspose
from math import ceil
from image_utils import load_images

# Keep deterministic results
from numpy.random import seed
from tensorflow import set_random_seed
seed(42)
set_random_seed(42)


def create_encoder(inputs: Input, filters_sizes: List[int], latent_dimension: int,
                   kernel_size=3) -> Model:
    x = inputs
    for filters in filters_sizes:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
    x = Flatten()(x)
    latent = Dense(latent_dimension, name='latent_vector')(x)

    encoder = Model(inputs, latent, name='encoder')
    return encoder


def calculate_last_conv_shape(input_shape: List[int],
                              filters_sizes: List[int]):
    x, y, _ = input_shape
    for filters in filters_sizes:
        x, y = ceil(x/2), ceil(y/2)
        f = filters

    return (x, y, f)


def create_decoder(last_conv_shape: List[int], filters_sizes: List[int],
                   latent_dimension: int, kernel_size=3) -> Model:
    latent_inputs = Input(shape=(latent_dimension,), name='decoder_input')
    x = Dense(
        last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]
    )(latent_inputs)
    x = Reshape(last_conv_shape)(x)
    for filters in filters_sizes[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
    outputs = Conv2DTranspose(filters=3,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder


def create_and_train_autoencoder():
    X = load_images()
    input_shape = X.shape[1:]
    inputs = Input(shape=input_shape, name='encoder_input')

    latent_dim = 2
    kernel_size = 5
    filters_sizes = [32, 64, 128, 256]

    divisible = 2 ** len(filters_sizes)
    assert input_shape[0] % divisible == 0, "Image x size should be divisible"
    assert input_shape[1] % divisible == 0, "Image y size should be divisible"

    encoder = create_encoder(inputs, filters_sizes,
                             latent_dim, kernel_size=kernel_size)

    last_conv_shape = calculate_last_conv_shape(input_shape, filters_sizes)
    decoder = create_decoder(last_conv_shape, filters_sizes,
                             latent_dim, kernel_size=kernel_size)

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    autoencoder.compile(loss='mse', optimizer='adam')
    autoencoder.fit(X,
                    X,
                    epochs=10,
                    batch_size=32)

    encoder.save(f"{TMP_DIR}/encoder.h5")
    decoder.save(f"{TMP_DIR}/decoder.h5")


if __name__ == "__main__":
    create_and_train_autoencoder()
