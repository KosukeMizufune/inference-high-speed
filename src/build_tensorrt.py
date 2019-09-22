import argparse

import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_model_path, engine_path):
    with trt.Builder(TRT_LOGGER) as builder:
        with builder.create_network() as network:
            with trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 30  # 1GB
                builder.max_batch_size = 1
                # fp16を用いる場合はコメントを外す
                # builder.fp16_mode = True
                with open(onnx_model_path, 'rb') as model:
                    parser.parse(model.read())
                engine = builder.build_cuda_engine(network)
                with open(engine_path, "wb") as f:
                    f.write(engine.serialize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--engine_path', type=str, default=None)

    args = parser.parse_args()

    build_engine(args.onnx_path, args.engine_path)
