import datetime
from typing import Optional
import base64

import torch
import yaml
import cv2
import fastapi
from fastapi.encoders import jsonable_encoder
import fastapi.middleware.cors
import numpy as np
from fastapi import Form, UploadFile, File
from pybsc.image_utils import decode_image
import pydantic
from pydantic import BaseModel
from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from pybsc.image_utils import masks_to_bboxes
from pybsc import checksum_md5
from pybsc.fastapi_utils import save_upload_file_to_tmp


device = 'cuda'
config_file = 'solo-foreground-2021-07-26/solo-foreground.py'
checkpoint_file = 'solo-foreground-2021-07-26/solo-foreground.pth'
model = init_detector(config_file,
                      checkpoint_file,
                      device=device)
model_md5 = checksum_md5(checkpoint_file)
fg_class_names_file = \
    'solo-foreground-2021-07-26/solo-foreground_classnames.yaml'
with open(fg_class_names_file, 'r') as f:
    fg_class_names = yaml.safe_load(f)['fg_class_names']

__version__ = '0.0.1'


class Image(BaseModel):
    data: Optional[str] = pydantic.Field(
        default=None, example=None,
        description='base64 encoded image')


class BodyExtract(BaseModel):
    image: Image
    score_thresh: Optional[float]


app = fastapi.FastAPI()
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_root():
    SERVICE = {
        "name": "solo",
        "version": __version__,
        "libraries": {
            "solo": __version__
        },
        'model_info': {
            'md5': model_md5,
        },
    }
    return {
        "service": SERVICE,
        "time": int(datetime.datetime.now().timestamp() * 1000),
    }


@app.post("/uploadfile")
async def create_upload_file(upload_file: UploadFile = File(...)):
    global model_md5
    model_path = save_upload_file_to_tmp(upload_file)
    new_model_md5 = checksum_md5(model_path)
    if new_model_md5 != model_md5:
        print('Update model({})'.format(model_md5))
        model.load_state_dict(torch.load(model_path)['state_dict'])
        model_md5 = new_model_md5


@app.post("/detect")
async def detect(data: BodyExtract):
    encoded_data = jsonable_encoder(data.image)
    image = decode_image(encoded_data['data'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    score_thresh = data.score_thresh or 0.2
    model.score_thresh = score_thresh
    result = inference_detector(model, image)
    h, w, _ = image.shape
    if result[0] is None:
        masks = np.zeros((0, h, w), dtype=np.int32)
        labels = np.zeros((0), dtype=np.int32)
        scores = np.zeros((0), dtype=np.float32)
    else:
        masks, labels, scores = result[0]
        masks = masks.cpu().numpy().astype(np.uint8)
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
    del result
    vis_inds = scores > score_thresh
    masks = masks[vis_inds]
    labels = labels[vis_inds]
    scores = scores[vis_inds]
    bboxes = masks_to_bboxes(masks)
    b64encoded_masks = base64.b64encode(
        masks.tobytes()).decode('ascii')

    SERVICE = {
        "name": "solo",
        "version": __version__,
        "libraries": {
            "solo": __version__
        },
        'model_info': {
            'md5': model_md5,
        },
    }
    return {
        "service": SERVICE,
        "time": int(datetime.datetime.now().timestamp() * 1000),
        'score_thresh': score_thresh,
        'class_names': fg_class_names,
        "response": {
            "width": image.shape[1],
            "height": image.shape[0],
            'labels': labels.tolist(),
            'label_names': [fg_class_names[i] for i in labels],
            'scores': scores.tolist(),
            'bboxes': bboxes.tolist(),
            'masks': b64encoded_masks,
        },
    }


@app.get('/info', tags=['Utility'])
def info():
    """Enslist container configuration.

    """
    about = dict(
        version=__version__,
    )
    return about
