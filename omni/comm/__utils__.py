from scipy.signal import upfirdn

from .__omni__ import *


# : normalize
# =====================================================================================================
def normalize(signal: array) -> array:
    return signal * np.sqrt(1.0 / np.mean(np.abs(signal) ** 2))

# def normalize(signal: np.ndarray) -> np.ndarray:
#     # 计算信号的均值
#     mean = np.mean(signal)
#     # 计算信号的标准差
#     std_dev = np.std(signal)
#     # 检查标准差是否为零，以避免除以零的错误
#     if std_dev == 0:
#         raise ValueError("Standard deviation of the signal is zero. Cannot perform Gaussian normalization.")
#     # 执行高斯归一化
#     normalized_signal = (signal - mean) / std_dev
#     return normalized_signal


# : rcosdesign
# =====================================================================================================
def rcosdesign(beta: float, span: int, sps: int, shape: str = 'sqrt') -> array:
    """Raised Cosine FIR Pulse-Shaping Filter Design"""
    eps = 2.2204E-16
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1, dtype=float) / sps
    h = np.zeros(span * sps + 1)
    if shape == 'sqrt':
        idx1 = np.where(t == 0)[0]
        if len(idx1) != 0:
            h[idx1] = -1.0 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)
        idx2 = np.where(np.abs(np.abs(4 * beta * t) - 1.0) < np.sqrt(eps))[0]
        if len(idx2) != 0:
            h[idx2] = 1.0 / (2.0 * np.pi * sps) * (np.pi * (beta + 1) * np.sin(
                np.pi * (beta + 1) / (4 * beta)) - 4 * beta * np.sin(
                np.pi * (beta - 1) / (4 * beta)) + np.pi * (beta - 1) * np.cos(
                np.pi * (beta - 1) / (4 * beta)))
        ind = np.arange(len(t))
        ind = np.delete(ind, np.concatenate((idx1, idx2)))
        nind = t[ind]
        h[ind] = -4 * beta / sps * (np.cos((1 + beta) * np.pi * nind) + np.sin(
            (1 - beta) * np.pi * nind) / (4 * beta * nind)) / (
                         np.pi * ((4 * beta * nind) ** 2 - 1))
    elif shape == 'normal':
        denom = 1 - (2 * beta * t) ** 2
        idx1 = np.where(np.abs(denom) > np.sqrt(eps))[0]
        h[idx1] = np.sinc(t[idx1]) * (
                np.cos(np.pi * beta * t[idx1]) / denom[idx1]) / sps
        idx2 = np.arange(len(t))
        idx2 = np.delete(idx2, idx1)
        h[idx2] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)
    else:
        info = 'filter shape is wrong'
        raise ValueError(info)
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def power(signal: array) -> float:
    return float(np.mean(np.abs(signal) ** 2))


__keep__ = [upfirdn]
