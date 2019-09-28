import argparse

from chainer import serializers, cuda
from chainercv.links import SSD300
from PIL import Image
import numpy as np

from local_lib.utils.time import stop_watch

log_filename = "chainer_ssd"


@stop_watch(log_filename)
def infer(img, model):
    _ = model.predict(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    model = SSD300(n_fg_class=20)
    model.use_preset('evaluate')
    if args.model_path.endswith('.npz'):
        serializers.load_npz(args.model_path, model)
    elif args.model_path:
        serializers.load_npz(args.model_path, model, 'updater/model:main/predictor/')
    else:
        raise ValueError('You must specify "model_path"')

    if args.gpu_id >= 0:
        cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu(args.gpu_id)

    img = Image.open(args.img_path)
    img = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    img = img[np.newaxis, ...]

    infer(img, model)
