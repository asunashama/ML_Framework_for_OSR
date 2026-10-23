from omni import *

from multiprocessing import Pool, Manager
from itertools import combinations as combination


# : hms
# ----------------------------------------------------------------------------------------------------
def hms(cost: float) -> str:
    hour, _ = np.divmod(cost, 3600)
    minute, second = np.divmod(cost - hour * 3600, 60)
    if hour > 0:
        return '{} h {} m {:>.2f} s'.format(int(hour), int(minute), second)
    elif hour == 0 and minute > 0:
        return '{} min : {:>.2f} sec'.format(int(minute), second)
    return '{:>.2f} sec'.format(second)


# : progress
# ----------------------------------------------------------------------------------------------------
def progress(p: float, cost: float):
    assert 0 <= p <= 1.0
    print('\r', end='')
    print('>>> {:.2%} | ETA = {}'.format(p, hms((1 - p) / p * cost)), end='')
    sys.stdout.flush()


# : mean
# ----------------------------------------------------------------------------------------------------
def mean(data: List[float]) -> float:
    mu = sum(data) / len(data)
    return mu


# : std
# ----------------------------------------------------------------------------------------------------
def std(data: List[float]) -> float:
    mu = mean(data)
    sigma = math.sqrt((sum([(x - mu) ** 2 for x in data])) / len(data))
    return sigma


# : Result Wrapper
# --------------------------------------------------------------------------------------------------
class Result(object):

    def __init__(self, data: List[float]):
        self._data = data
        self._mu = mean(data)
        self._sigma = std(data)
        self._hash = int(
            (str(self._mu) + str(max(data)) + str(min(data)) + str(self._sigma)).replace('.', ''))
        self._count = len(data)

    def __hash__(self):
        return self._hash

    @property
    def count(self) -> int:
        return self._count

    @property
    def data(self) -> list:
        return self._data

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma


# 1 'Num = (\s*)(\d+|0), Accuracy = (\d{1,2}\.\d{2})%'
# 2 'Num = (\s*)([0-9]\d*|0), Accuracy = (\d{1,2}\.\d{2})%'
def parse(line: str):
    if re.match(r'(\d{1,2})\s(\d{1,2}\.\d{2})%', line):
        accuracy = float(re.search(r'(\d{1,2}\.\d{2})', line).group())
        return accuracy


def check(data: List[float],
          items: List[Tuple[int, ...]],
          results: List[Result],
          name: Optional[str] = None,
          status: Optional[Dict[str, int]] = None):
    for k, item in enumerate(items):
        if name is not None:
            status[name] = k + 1
        target = [data[i] for i in item]
        results.append(Result(target))


def analyze(result: List[Result]):
    mu_sorted_index = sorted(range(len(result)),
                             key=lambda k: result[k].mu, reverse=True)
    sigma_sorted_index = sorted(range(len(result)),
                                key=lambda k: result[k].sigma)
    # print('count = {:>1d}, accuracy = {:2.2f} \u00B1 {:2.2%}'.format(
    #     result[mu_sorted_index[0]].count,
    #     result[mu_sorted_index[0]].mu,
    #     result[mu_sorted_index[0]].sigma / 100))
    # print('count = {:>1d}, accuracy = {:2.2f} \u00B1 {:2.2%}'.format(
    #     result[sigma_sorted_index[0]].count,
    #     result[sigma_sorted_index[0]].mu,
    #     result[sigma_sorted_index[0]].sigma / 100))

    print('{:2.2f}/{:2.2f}/{:>1d}'.format(
        result[mu_sorted_index[0]].mu,
        result[mu_sorted_index[0]].sigma,
        result[mu_sorted_index[0]].count))


def calibrate(path: str = 'result.txt',
              count: Optional[int] = None,
              use_multiprocessing: bool = True):
    data: List[float] = []
    with open(path, mode='r') as file:
        while line := file.readline():
            if accuracy := parse(line):
                data.append(accuracy)

    data = sorted(data, reverse=True)
    data = data[1:-1]

    if count is None:
        count = len(data) - 1

    items = []
    index = list(range(len(data)))
    items += list(combination(index, count))  # list all possible combinations

    results: List[Result] = []

    if use_multiprocessing:
        manager = Manager()
        status, results = manager.dict(), manager.list(results)
        start = time.perf_counter_ns()
        workload = len(items) // (os.cpu_count() // 2 - 1)
        num_of_tasks = int(math.ceil(len(items) / workload))
        pool = Pool(os.cpu_count() // 2 - 1)
        for k in range(num_of_tasks):
            name = f'Task {k}'
            status[name] = 0
            a = workload * k
            b = workload * (k + 1)
            if b > len(items):
                b = len(items)
            assigned_items = [items[i] for i in range(a, b)]
            pool.apply_async(check, (data, assigned_items, results, name, status))

        count_finished = 0
        while count_finished < len(items):
            count_finished = sum([status[task] for task in status])
            cost = (time.perf_counter_ns() - start) / 1E9
            p = count_finished / len(items)
            if count_finished == 0:
                p = 0.001
            # progress(p, cost)

        cost = (time.perf_counter_ns() - start) / 1E9

        # print('\r', end='')
        # print('{:.2%} | {} Executed ...\n'.format(1.0, hms(cost)))

        pool.close()
        pool.join()

        results = list(results)
    else:
        start = time.perf_counter_ns()

        check(data, items, results)

        cost = (time.perf_counter_ns() - start) / 1E9
        print('{:.2%} | {} Executed ...\n'.format(1.0, hms(cost)))

    if len(results) != 0:
        analyze(results)
    else:
        print('Cannot find it!!!')


def calibrate_v2(path: str = 'result.txt'):
    data: List[float] = []
    with open(path, mode='r') as file:
        while line := file.readline():
            if accuracy := parse(line):
                data.append(accuracy)

    data = sorted(data, reverse=True)
    data = data[1:-1]

    print('{:.2f}% \u00B1 {:.2f}'.format(mean(data), std(data)))


if __name__ == '__main__':
    localTime = time.strftime('%H:%M:%S', time.localtime())
    print(f'Analyzing at LocalTime {localTime}\n')
    calibrate('../export/result.txt', count=11)
    calibrate('../export/result.txt', count=10)
    calibrate('../export/result.txt', count=9)
    calibrate('../export/result.txt', count=8)
