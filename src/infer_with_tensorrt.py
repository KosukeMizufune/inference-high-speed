import argparse

from chainercv import transforms
import numpy as np
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

from local_lib import common
from local_lib.utils import stop_watch

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        with trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


@stop_watch
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
    context.execute(
        batch_size=batch_size, bindings=bindings
    )
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
    return [out.host for out in outputs], infer_time


def transform_img(img, insize=300):
    img = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    img = transforms.resize(img, (insize, insize))
    img -= np.array((123, 117, 104)).reshape((-1, 1, 1))
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)

    args = parser.parse_args()

    img = Image.open(args.img_path)
    img = transform_img(img)

    with load_engine(args.engine_path) as engine:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            inputs[0].host = img
            trt_outputs, infer_time = do_inference(
                context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
            )
