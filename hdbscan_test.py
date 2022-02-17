import pandas as pd
import numpy as np
import hdbscan

df = pd.DataFrame(np.array([
               [1185, 11, 1207, 29],
               [1186, 26, 1252, 15],
               [1164, 40, 1199, 49],
               [59, 117, 99, 155]
               ]),
               columns=['x_min', 'y_min', 'x_max', 'y_max'])

# clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.0, metric='euclidean', alpha=1.0, p=None,
#                  algorithm='best', leaf_size=1, cluster_selection_method='leaf', allow_single_cluster=True)

# clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, allow_single_cluster=True)  # [ 0 -1  0 -1]
# clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, allow_single_cluster=True) # [ 0  0  0 -1]


clusterer.fit(df)
cluster_labels = clusterer.labels_
print(cluster_labels)

