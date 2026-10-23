from .__data__ import Dataset
from omni import *


class Profiler(object):

    def __init__(self, model, dataset: Optional[Dataset] = None):
        self._model = model
        if dataset is None:
            data = np.random.randn(16 * 128, *model.input_shape)
            dataset = utils.Dataset(data)
        self._dataset = dataset

    @tf.function
    def _infer_on_(self, inputs):
        outputs = self._model(inputs)
        return outputs

    def run(self) -> float:
        result = []
        dataset = self._dataset.shuffle().batch(128)
        for _, samples in enumerate(dataset):
            self._infer_on_(utils.cast(samples))

        for _ in range(10):
            dataset = self._dataset.shuffle().batch(1)
            begin = time.perf_counter_ns()
            for _, samples in enumerate(dataset):
                self._infer_on_(utils.cast(samples))
            cost = (time.perf_counter_ns() - begin) / 1E6
            result.append(cost)
        result = sum(result) / 10 / (16 * 128)
        print('{:.3f} ms'.format(result))
        return result
