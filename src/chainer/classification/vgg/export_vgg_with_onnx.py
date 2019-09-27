import argparse

import chainer
from chainer import links as L
import onnx_chainer
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='models/vgg.onnx')

    args = parser.parse_args()

    model = L.VGG16Layers()

    img = np.zeros((3, 300, 300), dtype=np.float32)
    img = L.model.vision.vgg.prepare(img)
    img = img[np.newaxis, ...]

    chainer.config.train = False
    onnx_model = onnx_chainer.export(model, img, filename=args.onnx_path)
