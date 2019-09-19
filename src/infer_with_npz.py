import argparse

import chainer
from chainer import serializers
from chainercv.links import SSD300
from PIL import Image
import numpy as np

from utils import time_watch


@time_watch
def infer(model_path, img_path=None, gpu_id=0):
    model = SSD300(n_fg_class=20)
    model.use_preset('evaluate')
    if model_path.endswith('.npz'):
        serializers.load_npz(model_path, model)
    elif model_path:
        serializers.load_npz(model_path, model, 'updater/model:main/predictor/')
    else:
        raise ValueError('You must specify "model_path"')

    if gpu_id >= 0:
        chainer.cuda.get_device_from_id(0).use()
        model.to_gpu(gpu_id)

    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    img = img[np.newaxis, ...]

    _ = model(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    infer(args.model_path, args.img_path, args.gpu_id)
