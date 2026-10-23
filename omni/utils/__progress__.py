from .__data__ import Batch
from .__meta__ import Singleton
from .__omni__ import *


# ... History
# =====================================================================================================
class History(object):
    """
    Training History
    you can append a log like: history['loss'].append(loss)
    """

    def __init__(self, name: str, *others: str):
        self._log: Dict[str, List[float]] = {}
        for key in (name,) + others:
            self._log[key] = []

    def __getitem__(self, name: str) -> List[float]:
        return self._log[name]

    @property
    def epoch(self) -> List[int]:
        return [k for k in range(1, len(self._log['loss']) + 1)]


# ... State
# =====================================================================================================
class State(object):
    """
    Record losses or metrics during training.
    """

    as_decimal = '{:>.5f}'
    as_percentage = '{:>.2%}'

    def __init__(self, name: str, *, style: Optional[str] = None):
        self._name = name
        self._style = style if style else self.__style(name)
        self._record: Optional[List[float]] = None

    @staticmethod
    def __style(name: str) -> str:
        if name.find('accuracy') != -1:
            return State.as_percentage
        return State.as_decimal

    @property
    def name(self) -> str:
        return self._name

    @property
    def style(self) -> str:
        return self._style

    @property
    def value(self) -> float:
        if self._record is None:
            return math.nan
        return sum(self._record) / len(self._record)

    def __format__(self, format_spec) -> str:
        return f'{self._name} = {self._style.format(self.value)}'

    def __str__(self) -> str:
        return f'{self._name} = {self._style.format(self.value)}'

    def update(self, value: float):
        if self._record is None:
            self._record = []
        self._record.append(value)

    def rst(self):
        self._record = None


# ... Progress
# =====================================================================================================
class Progress(object, metaclass=Singleton):

    def __init__(self, name: str, *others: str):
        self._state: Dict[str, State] = {}
        self._traced: List[str] = []
        for _name_ in (name,) + others:
            if _name_.startswith('@'):
                _name_ = _name_.lstrip('@')
                self._traced.append(_name_)
            self._state[_name_] = State(_name_)
        self._history = History(*self._state.keys())
        self._begin = None
        self._step = 0
        self._stop = None
        self._epoch = 0

        self._batched_dataset: Optional[Batch] = None

    def bind(self, batched_dataset: Batch):
        self._batched_dataset = batched_dataset
        self._stop = len(batched_dataset)

        # RST ...
        self._step = 0
        self._begin = None
        for name in self._state:
            self._state[name].rst()
        return self

    @staticmethod
    def _hms(cost: float) -> str:
        hour = cost // 3600
        remainder = cost - hour * 60
        minute, second = remainder // 60, remainder % 60
        if hour > 0:
            return '{} h {} m {:>.2f} s'.format(int(hour), int(minute), second)
        elif hour == 0 and minute > 0:
            return '{} min : {:>.2f} sec'.format(int(minute), second)
        return '{:>.2f} sec'.format(second)

    def __call__(self, *, width: int = 110):
        cost = (time.perf_counter_ns() - self._begin) / 1E9
        print('\r', end='')
        print(
            'epoch {} | {} | cost = {}'.format(
                self._epoch + 1,
                ('{}, ' * len(self._state.keys())).format(*self._state.values())[:-2],
                self._hms(cost)))
        print('=' * width)

        for name in self._state:
            self._history[name].append(self._state[name].value)

        # RST ...
        self._step = 0
        self._begin = None
        for name in self._state:
            self._state[name].rst()

        self._epoch += 1

        self._batched_dataset.on_epoch_end()  # reset binded dataset on epoch end (just shuffle it ...)

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self._step == 0:
            self._begin = time.perf_counter_ns()

        _step = self._step

        self._step += 1

        if self._step > 1:
            cost = (time.perf_counter_ns() - self._begin) / 1E9

            p = self._step / self._stop

            states = [self._state[name] for name in self._traced]

            print('\r', end='')
            print('epoch {} | {} | >>> {:.2%} | ETA = {}'.format(
                self._epoch + 1,
                ('{}, ' * len(states)).format(*states)[:-2],
                p, self._hms((1.0 - p) / p * cost)),
                end='')
            sys.stdout.flush()

        if self._step == self._stop:
            raise StopIteration

        return _step

    def update(self, name: str, value: float):
        self._state[name].update(value)

    @property
    def history(self) -> History:
        return self._history
