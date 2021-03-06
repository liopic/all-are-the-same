import requests
import os
from glob import glob
from collections import Counter
from PIL import Image
from config import IMG_URL, TMP_DIR, MEMBERS, LEGISLATURA


def download_and_save_img(member_id: int, legislatura_id: int):
    path = f"{TMP_DIR}/{member_id}_{legislatura_id}.jpg"
    if os.path.isfile(path):
        return

    r = requests.get(IMG_URL % (member_id, legislatura_id), stream=True)
    if r.status_code == 200:
        img = r.raw.read()
        with open(path, 'wb') as f:
            f.write(img)


def uniform_images(path: str):
    widths = Counter()
    heights = Counter()
    for file in glob(f"{path}/*.jpg"):
        i = Image.open(file)
        (w, h) = i.size
        widths[w] += 1
        heights[h] += 1

    most_common_size = (
        widths.most_common()[0][0],
        heights.most_common()[0][0]
    )
    print(f'Original most_common_size: {most_common_size}')

    most_common_size = (
        _find_16_divisible(most_common_size[0]),
        _find_16_divisible(most_common_size[1])
    )
    print(f'Adapted most_common_size: {most_common_size}')

    for file in glob(f"{path}/*.jpg"):
        i = Image.open(file)
        if len(i.getbands()) != 3:
            i = i.convert('RGB')
        resized = i.resize(most_common_size, Image.BICUBIC)
        resized.save(file, "JPEG")


def _find_16_divisible(x):
    return int(x/16) * 16


if __name__ == "__main__":
    if not os.path.isdir(TMP_DIR):
        os.mkdir(TMP_DIR)

    print(f'Downloading diputados images in {TMP_DIR}')
    for member_id in range(MEMBERS):
        download_and_save_img(member_id+1, LEGISLATURA)

    print('Rescaling images to most common size')
    uniform_images(TMP_DIR)
