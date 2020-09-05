import numpy as np
from tensorflow.keras.models import Model, load_model
from config import TMP_DIR, GENERATED_IMAGES_DIR
from image_utils import load_images, save_image

NUMBER_OF_IMAGES_FOR_MAP = (20, 14)
IMAGES_SIZE = (112, 160)


def create_a_map_of_even_separated_points(positions):

    def calculate_limits_covering_most_cases(positions):
        q01 = np.quantile(positions, 0.01, axis=0)
        q99 = np.quantile(positions, 0.99, axis=0)
        return (q01, q99)

    def build_even_separated_points(limits,
                                    number_of_items=NUMBER_OF_IMAGES_FOR_MAP):
        (q01, q99) = limits
        x_points = np.linspace(q01[0], q99[0], number_of_items[0])
        y_points = np.linspace(q01[1], q99[1], number_of_items[1])
        return (x_points, y_points)

    limits = calculate_limits_covering_most_cases(positions)
    return build_even_separated_points(limits)


def build_latent_space_map(points, decoder: Model):
    (x_points, y_points) = points
    (width, height) = IMAGES_SIZE
    image_size_x = width * len(x_points)
    image_size_y = height * len(y_points)
    image = np.zeros((image_size_y, image_size_x, 3))

    for step_x, x in enumerate(x_points):
        for step_y, y in enumerate(y_points):
            [decoded] = decoder.predict(np.array([[x, y]]))

            pixel_x = step_x * width
            pixel_y = step_y * height
            image[pixel_y:pixel_y+height, pixel_x:pixel_x+width] = decoded

    return image


if __name__ == "__main__":
    encoder = load_model(f"{TMP_DIR}/encoder_13.h5", compile=False)
    decoder = load_model(f"{TMP_DIR}/decoder_13.h5", compile=False)

    images = load_images()
    predicted_positions = encoder.predict(images)

    points = create_a_map_of_even_separated_points(predicted_positions)

    image = build_latent_space_map(points, decoder)
    filepath = f"{GENERATED_IMAGES_DIR}/latent_map.png"
    save_image(image, filepath)
    print(f"saved at {filepath}")
