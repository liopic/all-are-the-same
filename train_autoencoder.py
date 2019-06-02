from glob import glob
from PIL import Image
import numpy as np
from typing import List
from config import TMP_DIR
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Reshape, Conv2DTranspose
from math import ceil


def load_images() -> np.array:
    files = glob(f"{TMP_DIR}/*.jpg")
    images = []
    for file in files:
        image_data = np.asarray(Image.open(file))
        # normalize to range [0, 1], from [0, 255]
        images.append(image_data.astype('float32') / 255)

    ret = np.asarray(images)
    return ret


def encoder(inputs: Input, filters: List[int], latent_dimension: int,
            kernel_size=3) -> Model:
    x = inputs
    for filters in filters_sizes:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    encoder = Model(inputs, latent, name='encoder')
    return encoder


def calculate_last_conv_shape(input_shape: List[int], filters: List[int]):
    x, y, _ = input_shape
    for filters in filters_sizes:
        x, y = ceil(x/2), ceil(y/2)
        f = filters

    return (x, y, f)


def decoder(last_conv_shape: List[int], filters: List[int],
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


if __name__ == "__main__":
    X = load_images()
    input_shape = X.shape[1:]
    inputs = Input(shape=input_shape, name='encoder_input')

    latent_dim = 2
    kernel_size = 5
    filters_sizes = [32, 64, 128, 256]

    divisible = 2 ** len(filters_sizes)
    assert input_shape[0] % divisible == 0, "Image x size should be divisible"
    assert input_shape[1] % divisible == 0, "Image y size should be divisible"

    encoder = encoder(inputs, filters_sizes, latent_dim)

    last_conv_shape = calculate_last_conv_shape(input_shape, filters_sizes)
    decoder = decoder(last_conv_shape, filters_sizes, latent_dim)

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    autoencoder.compile(loss='mse', optimizer='adam')
    history = autoencoder.fit(X,
                              X,
                              epochs=10,
                              batch_size=32)

    encoder.save(f"{TMP_DIR}/encoder.h5")
    decoder.save(f"{TMP_DIR}/decoder.h5")
