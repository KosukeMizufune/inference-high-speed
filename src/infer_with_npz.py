import chainer
from chainer import serializers, iterators
from chainercv.links import SSD300
from chainercv.evaluations import eval_detection_voc
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets.voc.voc_bbox_dataset import VOCBboxDataset


def infer(model_path, data_dir=None, gpu_id=0):
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

    test = VOCBboxDataset(year='2007', split='test', use_difficult=True, return_difficult=True)
    test_iter = iterators.SerialIterator(test, 32, repeat=False, shuffle=False)
    evaluator = DetectionVOCEvaluator(test_iter, model, use_07_metric=True, label_names=voc_bbox_label_names)
    print(evaluator.evaluate())


if __name__ == "__main__":
    infer('../models/snapshot_ssd', gpu_id=-1)
