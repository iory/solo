import base64
import json
from typing import List

import requests
from pybsc.image_utils import encode_image
import cv2


def inference_solo(
        img,
        url: str = 'http://localhost:8061/detect/'):
    response = requests.post(
        url,
        json={'image':
              {'data': encode_image(img)}
        })
    data = response.json()
    return data


img = cv2.imread('./demo.jpg')
res = inference_solo(img)
