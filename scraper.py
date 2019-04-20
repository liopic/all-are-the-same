import requests
import os
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


if __name__ == "__main__":
    if not os.path.isdir(TMP_DIR):
        os.mkdir(TMP_DIR)
    print(f'Downloading diputados images in {TMP_DIR}')
    for member_id in range(MEMBERS+1):
        download_and_save_img(member_id, LEGISLATURA)
