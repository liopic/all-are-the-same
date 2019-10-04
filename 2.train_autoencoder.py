from typing import List
from config import TMP_DIR, LEGISLATURA
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Reshape, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from math import ceil
from image_utils import load_images
from config import KERNEL_SIZE, FILTER_SIZES, EPOCHS, BATCH_SIZE

# Keep deterministic results
from numpy.random import seed
from tensorflow import set_random_seed
seed(42)
set_random_seed(42)


def create_and_train_autoencoder():
    latent_dim = 2
    kernel_size = KERNEL_SIZE
    filters_sizes = FILTER_SIZES
    epochs = EPOCHS
    batch_size = BATCH_SIZE

    X = load_images()
    input_shape = X.shape[1:]
    inputs = Input(shape=input_shape, name='encoder_input')

    divisible = 2 ** len(filters_sizes)
    assert input_shape[0] % divisible == 0, "Image x size should be divisible"
    assert input_shape[1] % divisible == 0, "Image y size should be divisible"

    encoder = _create_encoder(inputs, filters_sizes,
                              latent_dim, kernel_size=kernel_size)

    last_conv_shape = _calculate_last_conv_shape(input_shape, filters_sizes)
    decoder = _create_decoder(last_conv_shape, filters_sizes,
                              latent_dim, kernel_size=kernel_size)

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    optimizer = Adam(lr=0.0005)
    autoencoder.compile(loss='mse', optimizer=optimizer)
    history = autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size)

    encoder.save(f"{TMP_DIR}/encoder_{LEGISLATURA}-{EPOCHS}-{KERNEL_SIZE}.h5")
    decoder.save(f"{TMP_DIR}/decoder_{LEGISLATURA}-{EPOCHS}-{KERNEL_SIZE}.h5")
    loss_file = f"{TMP_DIR}/loss_{LEGISLATURA}-{EPOCHS}-{KERNEL_SIZE}.txt"
    with open(loss_file, 'w') as f:
        for loss in history.history['loss']:
            f.write(f"{loss}\n")


def _create_encoder(inputs: Input, filters_sizes: List[int],
                    latent_dimension: int, kernel_size=3) -> Model:
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


def _calculate_last_conv_shape(input_shape: List[int],
                               filters_sizes: List[int]):
    x, y, _ = input_shape
    for filters in filters_sizes:
        x, y = ceil(x/2), ceil(y/2)
        f = filters

    return (x, y, f)


def _create_decoder(last_conv_shape: List[int], filters_sizes: List[int],
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
    print("Using the following device:")
    print(K.tensorflow_backend._get_available_gpus())

    create_and_train_autoencoder()
