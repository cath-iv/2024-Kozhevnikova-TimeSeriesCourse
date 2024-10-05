import numpy as np
import math


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    ed_dist = 0

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def const_find(a):
    """Вычисляет среднее и стандартное отклонение."""
    mean = np.mean(a)
    std_dev = np.std(a, ddof=0)  # Полное стандартное отклонение
    return mean, std_dev


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Рассчитывает нормализованное евклидово расстояние с учетом среднем и стандартного отклонения.

    Параметры
    ----------
    ts1: первый временной ряд
    ts2: второй временной ряд

    Возвращает
    -------
    norm_ed_dist: нормализованное евклидово расстояние между ts1 и ts2
    """
    n = len(ts1)
    mean1, std1 = const_find(ts1)
    mean2, std2 = const_find(ts2)

    # Вычисление скалярного произведения
    spvr = np.sum(ts1 * ts2)

    # Нормализованное значение D
    D = (spvr - n * mean1 * mean2) / (n * std1 * std2)

    # Возврат нормализованного евклидова расстояния
    return math.sqrt(abs(2 * n * (1 - D)))


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance with Sakoe-Chiba band

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size (percentage of the length of the time series)

    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """
    n = len(ts1)
    m = len(ts2)

    d = np.inf * np.ones((n + 1, m + 1))
    d[0][0] = 0
    r = int(r * n)

    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(m + 1, i + r + 1)):
            cost = np.power((ts1[i - 1] - ts2[j - 1]), 2)
            d[i][j] = cost + np.min([d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]])

    return d[n][m]
