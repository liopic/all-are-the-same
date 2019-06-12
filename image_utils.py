from glob import glob
from PIL import Image
import numpy as np
from config import TMP_DIR


def load_images() -> np.array:
    files = glob(f"{TMP_DIR}/*.jpg")
    images = []
    for file in files:
        image_data = np.asarray(Image.open(file))
        # normalize to range [0, 1], from [0, 255]
        images.append(image_data.astype('float32') / 255)

    ret = np.asarray(images)
    return ret
