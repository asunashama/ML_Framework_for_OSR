from collections import abc

from .__omni__ import *


# : split
# ==========================================================================================================
def split(source: Dict[str, array], how: Optional[Tuple[int, ...]] = None):
    def _size_of_(source: Dict[str, array]) -> int:
        num_of_samples = len(source[list(source.keys())[-1]])
        assert [num_of_samples == len(source[name])
                for name in source].count(True) == len(source)
        return num_of_samples

    if how is None:
        num_of_samples = _size_of_(source)
        half = int(num_of_samples // 2)
        how = (half, num_of_samples - half)
    result: List[Dict[str, array]] = []
    for k in range(len(how)):
        beg = sum(how[:k])
        end = sum(how[:k + 1])
        cache: Dict[str, array] = {}
        for name in source:
            cache[name] = source[name][beg:end]
        result.append(cache)
    return result


# : shuffle
# ==========================================================================================================
@overload
def shuffle(source: Dict[str, array]):
    pass


@overload
def shuffle(data: array, *others: array):
    pass


@dispatch
def shuffle(*args, **kwargs):
    shuffle(*args, **kwargs)


@shuffle.register(dict)
def _(source: Dict[str, array]):
    for name in source:
        np.random.shuffle(source[name])
    return source


@shuffle.register(array)
def _(data: array, *others: array):
    assert [len(data) == len(cache)
            for cache in others].count(True) == len(others)
    new = np.random.permutation(len(data))
    for cache in (data,) + others:
        cache[:] = cache[new]
    if others:  # others are not None ...
        return (data,) + others
    return data


# : categorical
# ==========================================================================================================
def to_categorical(labels: array, num_of_classes: Optional[int] = None) -> array:
    """labels should be of shape (N, ) or (N, 1)"""
    result = np.zeros((len(labels), np.max(labels) + 1 if num_of_classes is None else num_of_classes))
    result[np.arange(len(labels)), labels.astype(int).flatten()] = 1.0
    return result


# : make
# ==========================================================================================================
def make(source: Dict[str, array], *,
         labeled: bool = True, encoded: bool = True,
         use: Optional[Callable[array, array]] = None) -> array | Tuple[array, array]:
    if labeled:
        """Needs to Label It"""
        samples, labels = [], []
        for k, name in enumerate(source):
            samples.append(source[name])
            labels.append(np.ones((len(source[name]), 1)) * k)
        samples, labels = np.vstack(samples), np.vstack(labels)
        if encoded:
            labels = to_categorical(labels, len(source))
        if use is not None:
            samples = use(samples)
        return samples, labels
    samples = np.vstack([source[name] for name in source])
    if use is not None:
        samples = use(samples)
    return samples

# : make_Gai
# ==========================================================================================================
def make_Gai(source: Dict[str, array], *,
         labeled: bool = True, encoded: bool = True,
         use: Optional[Callable[array, array]] = None) -> array | Tuple[array, array]:
    if labeled:
        """Needs to Label It"""
        samples, samples_inner, labels ,labels_inner = [], [], [], []
        for k, name in enumerate(source): # enumerate函数返回 index 和 值
            samples.append(source[name])
            #labels.append(np.ones((len(source[name]), 1)) * k) # 真实标签
            for j, inner_name in enumerate(source[name]): #
                samples_inner.append(samples[k][inner_name])
                labels.append(np.ones((len(samples[k][inner_name]), 1)) * k)
                labels_inner.append(np.ones((len(samples[k][inner_name]), 1)) * ((int(inner_name)-1) + k * 100))

        samples, labels, labels_inner = np.vstack(samples_inner), np.vstack(labels) ,np.vstack(labels_inner)
        if encoded:
            labels = to_categorical(labels, len(source))
        if use is not None:
            samples = use(samples)
        return samples, labels, labels_inner
    samples = np.vstack([source[name] for name in source])
    if use is not None:
        samples = use(samples)
    return samples


# : archive
# ==========================================================================================================
def archive(samples: array, labels: array, *,
            category: Optional[list[str]] = None, num_of_classes: int = 10) -> dict[str, array]:
    source: dict[str, array] = {}
    if category is None:
        category = [f'C{k + 1}' for k in range(num_of_classes)]
    labels = labels.flatten()
    for k, name in enumerate(category):
        indices = np.where(labels == k)
        source[name] = samples[indices]
    return source


# : cast
# ==========================================================================================================
def cast(data: array, *others: array) -> Tensor | Tuple[Tensor, ...]:
    """cast from Array to Tensor"""
    if others:  # if others are not None ...
        return tuple(tf.cast(cache, tf.float32) for cache in (data,) + others)
    return tf.cast(data, tf.float32)


# ... Dataset
# ==========================================================================================================
class Dataset(abc.Sequence):

    def __init__(self, data: array, *others: array):
        self._num_of_samples = len(data)
        if others:
            for _data_ in others:
                assert len(_data_) == self._num_of_samples
            self._storage: List[array, ...] = [_data_.copy() for _data_ in (data,) + others]
            self._num_of_members = len(self._storage)
        else:
            self._storage: array = data.copy()
            self._num_of_members = 1

    def __iter__(self):
        return Iterator(self)

    def __getitem__(self, key: Any) -> array | Tuple[array, ...]:
        if self._num_of_members > 1:
            return tuple(data[key] for data in self._storage)
        return self._storage[key]

    def __len__(self) -> int:
        return self._num_of_samples

    def shuffle(self):
        new = np.random.permutation(self._num_of_samples)
        if self._num_of_members > 1:
            for _data_ in self._storage:
                _data_[:] = _data_[new]
        else:
            self._storage[:] = self._storage[new]
        return self

    def batch(self, size: int = 128):
        return Batch(self, size=size)


# ... Batch
# ==========================================================================================================
class Batch(keras.utils.Sequence):

    def __init__(self, dataset: Dataset, *, size: int = 128):
        self._dataset = dataset
        self._total = int(math.ceil(len(dataset) / size))
        self._loc = np.arange(len(dataset))
        self._size = size

        self._index = 0  # only for infinitely get

    def on_epoch_end(self):
        np.random.shuffle(self._loc)

    def __iter__(self):
        return Iterator(self)

    def __getitem__(self, index: int) -> array | Tuple[array, ...]:
        beg = index * self._size
        end = (index + 1) * self._size
        batch = self._dataset[self._loc[beg:end]]
        return batch

    def __len__(self) -> int:
        return self._total

    def shuffle(self):
        np.random.shuffle(self._loc)
        return self

    def fetch(self) -> array | Tuple[array, ...]:
        """randomly get a batch in infinite mode"""
        if self._index == self._total:
            np.random.shuffle(self._loc)
            self._index = 0
        beg = self._index * self._size
        end = (self._index + 1) * self._size
        self._index += 1
        batch = self._dataset[self._loc[beg:end]]
        return batch


# ... Iterator
# ==========================================================================================================
class Iterator(abc.Iterator):

    def __next__(self) -> array | Tuple[array, ...]:
        if self._count == len(self._target):
            if isinstance(self._target, Batch):
                self._target.on_epoch_end()
            raise StopIteration
        value = self._target[self._count]
        self._count += 1
        return value

    def __iter__(self):
        return self

    def __init__(self, target: Dataset | Batch):
        self._target = target
        self._count = 0
