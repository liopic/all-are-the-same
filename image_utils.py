from PIL import Image
import numpy as np
from config import TMP_DIR, LEGISLATURA, MEMBERS


def load_images() -> np.array:
    files = [f"{TMP_DIR}/{i+1}_{LEGISLATURA}.jpg" for i in range(MEMBERS)]
    images = []
    for file in files:
        image_data = np.asarray(Image.open(file))
        # normalize to range [0, 1], from [0, 255]
        images.append(image_data.astype('float32') / 255)

    ret = np.asarray(images)
    return ret

def save_image(image_array, filename):
    rgb = image_array*255
    image = Image.fromarray(np.uint8(rgb), 'RGB')
    image.save(filename)
