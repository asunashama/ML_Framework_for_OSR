from .__omni__ import *
from .__progress__ import History


# ... EarlyStopping
# =====================================================================================================
class EarlyStopping(object):

    def __init__(self, history: History):
        self._history = history
        self._cntNaN = 0
        self._cntMax = 0
        self._flag = False

    def __call__(self) -> bool:
        if math.isnan(self._history['val_loss'][-1]):
            self._cntNaN += 1
            if self._cntNaN >= 10:
                self._flag = True

        if self._history['accuracy'][-1] == 1.00:
            self._cntMax += 1
            if self._cntMax >= 50:
                self._flag = True

        return self._flag


# ...
def write_HDF(path: str, dataset: Dict[str, array]):
    with HDF.File(path, 'w') as file:
        for name, data in dataset.items():
            file.create_dataset(name, data=np.asarray(data))
