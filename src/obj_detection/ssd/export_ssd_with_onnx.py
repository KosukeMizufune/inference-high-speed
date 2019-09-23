import argparse

import chainer
from chainer import serializers
from chainercv.links import SSD300
import onnx_chainer
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--onnx_path', type=str, default='models/ssd.onnx')

    args = parser.parse_args()

    model = SSD300(n_fg_class=20)
    model.use_preset('evaluate')
    if args.model_path.endswith('.npz'):
        serializers.load_npz(args.model_path, model)
    elif args.model_path:
        serializers.load_npz(args.model_path, model, 'updater/model:main/predictor/')
    else:
        raise ValueError('You must specify "model_path"')

    x = np.zeros((1, 3, 300, 300), dtype=np.float32)

    chainer.config.train = False
    onnx_model = onnx_chainer.export(model, x, filename=args.onnx_path)
