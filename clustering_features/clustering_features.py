import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

number_of_clusters = 30

print('Started feature clustering script')

feature_arrays = pd.read_csv('../features/opencv_features.csv', sep=',')

feacols = [col for col in feature_arrays.columns if col not in ['img','classname']]

print('Starting to cluster')

estimator = MiniBatchKMeans(n_clusters=number_of_clusters, init='random')#, n_jobs=-2)

cluster = estimator.fit(feature_arrays[feacols].values)

print('features are clustered')

feature_arrays["cluster_labels"] = cluster.labels_

feature_arrays.to_csv("../features/clustered_features_" + str(number_of_clusters) + ".csv", index=None)

print('Clustered features written to the file')


