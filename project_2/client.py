import os
from argparse import ArgumentParser

import requests

parser = ArgumentParser()
parser.add_argument("--image", type=str, help="Path of the image to be classified")

args = parser.parse_args()

if not os.path.exists(args.image):
    raise ValueError("Image does not exist")

URL = "http://0.0.0.0:8000/get_car_angle"
IMAGE_NAME = args.image

with open(IMAGE_NAME, 'rb') as f:
    files = {"file": (IMAGE_NAME, f)}
    response = requests.post(URL, files=files)
    print(response.json())
