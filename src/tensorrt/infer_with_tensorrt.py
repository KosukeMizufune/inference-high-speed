import argparse

from chainercv import transforms
import numpy as np
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

from local_lib.tensorrt import common
from local_lib.utils.utils import stop_watch

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
log_filename = "tensorrt_vgg"


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        with trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


@stop_watch(log_filename)
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
    context.execute(
        batch_size=batch_size, bindings=bindings
    )
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
    return [out.host for out in outputs]


def transform_img(img, img_size):
    img = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    img = transforms.resize(img, (img_size, img_size))
    img -= np.array((123, 117, 104)).reshape((-1, 1, 1))
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=224)

    args = parser.parse_args()

    img = Image.open(args.img_path)
    img = transform_img(img, args.img_size)

    with load_engine(args.engine_path) as engine:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            inputs[0].host = img
            trt_outputs = do_inference(
                context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
            )
