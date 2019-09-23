import argparse

from chainer import cuda
from chainer import links as L
from PIL import Image
import numpy as np

from local_lib.utils import stop_watch


@stop_watch
def infer(img, model):
    _ = model(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    model = L.VGG16Layers()

    if args.gpu_id >= 0:
        cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu(args.gpu_id)

    img = Image.open(args.img_path)
    img = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    img = L.model.vision.vgg.prepare(img)
    img = img[np.newaxis, ...]

    infer(img, model)
