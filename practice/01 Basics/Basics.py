import os
import numpy as np
import random
import plotly.io as pio

from sktime.distances import euclidean_distance, dtw_distance, pairwise_distance
from sklearn.metrics import silhouette_score
import cv2
import imutils
import glob
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
import cv2

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.pairwise_distance import PairwiseDistance
from modules.clustering import TimeSeriesHierarchicalClustering
#from modules.classification import TimeSeriesKNN, calculate_accuracy
#from modules.image_converter import image2ts
from modules.utils import read_ts, z_normalize, sliding_window, random_walk
from modules.plots import plot_ts


practice_dir_path = r'C:\Users\79634\Documents\Универ\Временные ряды\GIT\practice\01 Basics'
os.chdir(practice_dir_path)
# %load_ext autoreload
# %autoreload 2

print('Задача 1')
def test_distances(dist1: float, dist2: float) -> None:
    """
    Check whether your distance function is implemented correctly

    Parameters
    ----------
    dist1 : distance between two time series calculated by sktime
    dist2 : distance between two time series calculated by your function
    """

    np.testing.assert_equal(round(dist1, 5), round(dist2, 5), 'Distances are not equal')

ts1 = random_walk(10)
ts2 = random_walk(10)
test1 = euclidean_distance(ts1, ts2)
test2 = ED_distance(ts1, ts2)
print(test1)
print(test2)
print(test_distances(test1, test2))

print("Задача 2")

test1 = dtw_distance(ts1, ts2)
test2 = DTW_distance(ts1, ts2)
print(test1)
print(test2)
test_distances(test1, test2)

print("Задача 3")

def test_matrices(matrix1 : np.ndarray, matrix2 : np.ndarray) -> None:
    """
    Check whether your matrix function is implemented correctly

    Parameters
    ----------
    matrix1 : distance matrix calculated by sktime
    matrix2 : distance matrix calculated by your function
    """

    np.testing.assert_equal(matrix1.round(5), matrix2.round(5), 'Matrices are not equal')


K = 5
n = 100
time_series = np.random.normal(loc=0, scale=1, size=(K, n))
sktime_euclidean_distances = pairwise_distance(time_series, metric='euclidean')
sktime_dtw_distances = pairwise_distance(time_series, metric="dtw")

pairwise_distance_calculator1 = PairwiseDistance(metric='dtw')
dtw_distances = pairwise_distance_calculator1.calculate(time_series)

pairwise_distance_calculator2 = PairwiseDistance(metric='euclidean')
euclidean_distances = pairwise_distance_calculator2.calculate(time_series)
print("Евклидова метрика sktime: ", sktime_euclidean_distances,", рассчитанная: ", euclidean_distances)
print("DTW метрика sktime: ", sktime_dtw_distances,", рассчитанная: ", dtw_distances)
test_matrices1 = test_matrices(sktime_euclidean_distances, euclidean_distances)
test_matrices2 = test_matrices(sktime_dtw_distances, dtw_distances)

print("Задача 4")

import pandas as pd

url = 'https://raw.githubusercontent.com/cath-iv/2024-Kozhevnikova-TimeSeriesCourse/main/practice/01%20Basics/datasets/part1/CBF_TRAIN.txt'

#data = read_ts(url)
data = pd.read_csv(url, delim_whitespace=True)

ts_set = data.iloc[:, 1:].values
labels = data.iloc[:, 0]

#plot_ts(ts_set)

# Создаем экземпляр класса PairwiseDistance для вычисления матриц расстояний
pairwise_distance = PairwiseDistance()

# 1. Кластеризация с использованием евклидовой метрики
# Вычисляем матрицу расстояний
eu_pairwise_distance_calculator = PairwiseDistance(metric='euclidean')
euclidean_distance_matrix = eu_pairwise_distance_calculator.calculate(ts_set)

# Выполняем кластеризацию
hierarchical_clustering_euclidean = TimeSeriesHierarchicalClustering(n_clusters=3, method='complete')
eu_labels = hierarchical_clustering_euclidean.fit_predict(euclidean_distance_matrix)

# Визуализируем результаты в виде дендрограммы
#hierarchical_clustering_euclidean.plot_dendrogram(ts_set, eu_labels, ts_hspace=5, title='Дендрограмма EU')

# 2. Кластеризация с использованием DTW
# Вычисляем матрицу расстояний для DTW
dtw_pairwise_distance_calculator = PairwiseDistance(metric='dtw')
dtw_distance_matrix = dtw_pairwise_distance_calculator.calculate(ts_set)

# Выполняем кластеризацию
hierarchical_clustering_dtw = TimeSeriesHierarchicalClustering(n_clusters=3, method='complete')
dtw_labels = hierarchical_clustering_dtw.fit_predict(dtw_distance_matrix)

# Визуализируем результаты в виде дендрограммы

#hierarchical_clustering_dtw.plot_dendrogram(ts_set, dtw_labels, ts_hspace=5, title='Дендрограмма DTW')

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]

for n_clusters in range_n_clusters:
    hierarchical_clustering_euclidean = TimeSeriesHierarchicalClustering(n_clusters=n_clusters, method='average')
    eu_labels = hierarchical_clustering_euclidean.fit_predict(euclidean_distance_matrix)
    hierarchical_clustering_dtw = TimeSeriesHierarchicalClustering(n_clusters=n_clusters, method='average')
    dtw_labels = hierarchical_clustering_dtw.fit_predict(dtw_distance_matrix)
    silhouette_DTW = silhouette_score(ts_set, dtw_labels)
    silhouette_EU = silhouette_score(ts_set, eu_labels)
    print("For DTW n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_DTW)
    print("For EU n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_EU)
    print("------------------------------------------------------------------------------------")

print("Задача 5")

ts1 = random_walk(10)
ts2 = random_walk(10)
from sktime.distances import euclidean_distance
ts1_norm = z_normalize(ts1)
ts2_norm = z_normalize(ts2)
test1 = euclidean_distance(ts1_norm, ts2_norm)
test2 = norm_ED_distance(ts1, ts2)
print(test1)
print(test2)
print(test_distances(test1, test2))

print("Задача 6")

url1 = 'https://raw.githubusercontent.com/cath-iv/2024-Kozhevnikova-TimeSeriesCourse/refs/heads/main/practice/01%20Basics/datasets/part2/chf10.csv'
ts1 = read_ts(url1)

url2 = 'https://raw.githubusercontent.com/cath-iv/2024-Kozhevnikova-TimeSeriesCourse/refs/heads/main/practice/01%20Basics/datasets/part2/chf11.csv'
ts2 = read_ts(url2)
ts_set = np.concatenate((ts1, ts2), axis=1).T
#plot_ts(ts_set)
m = 125
subs_set1 = sliding_window(ts_set[0], m, m-1)
subs_set2 = sliding_window(ts_set[1], m, m-1)
subs_set = np.concatenate((subs_set1[0:15], subs_set2[0:15]))
labels = np.array([0]*subs_set1[0:15].shape[0] + [1]*subs_set2[0:15].shape[0])

pairwise_distance = PairwiseDistance()

# 1. Кластеризация с использованием евклидовой метрики
# Вычисляем матрицу расстояний без нормализации

euclidean_distance_matrix = PairwiseDistance(metric='euclidean', is_normalize=False).calculate(subs_set)
hierarchical_clustering_euclidean = TimeSeriesHierarchicalClustering(n_clusters=2, method='complete').fit(euclidean_distance_matrix)
#hierarchical_clustering_euclidean.plot_dendrogram(subs_set, labels, ts_hspace=3, title='Дендрограмма 1')

# 2. Кластеризация с использованием нормализованной евклидовой метрики
# Вычисляем матрицу расстояний

norm_euclidean_distance_matrix = PairwiseDistance(metric='euclidean', is_normalize = True).calculate(subs_set)
hierarchical_clustering_euclidean_norm = TimeSeriesHierarchicalClustering(n_clusters=3, method='complete').fit(norm_euclidean_distance_matrix)
#hierarchical_clustering_euclidean_norm.plot_dendrogram(subs_set, labels, ts_hspace=3, title='Дендрограмма 2')

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
  hierarchical_clustering_euclidean = TimeSeriesHierarchicalClustering(n_clusters=n_clusters, method='complete')
  eu_labels = hierarchical_clustering_euclidean.fit_predict(euclidean_distance_matrix)
  hierarchical_clustering_euclidean_norm = TimeSeriesHierarchicalClustering(n_clusters=n_clusters, method='complete')
  norm_eu_labels = hierarchical_clustering_euclidean_norm.fit_predict(norm_euclidean_distance_matrix)
  silhouette_EU = silhouette_score(subs_set, eu_labels)
  silhouette_EU_N = silhouette_score(subs_set, norm_eu_labels)
  print("For EU n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_EU)
  print("For EU normalised n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_EU_N)
  print("------------------------------------------------------------------------------------")

print("Задача 7")

