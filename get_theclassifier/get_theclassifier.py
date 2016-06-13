import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

number_of_clusters = 30

cluster_vectors = pd.read_csv('../features/cluster_vectors_' + str(number_of_clusters) + '.csv', sep=',')

cols = [col for col in cluster_vectors.columns if col not in ['img','classname']]

theclassifier = LogisticRegression(multi_class='ovr', max_iter=1000, n_jobs=-2)

theclassifier.fit(cluster_vectors[cols].values, cluster_vectors.classname.values)

with open('../classifiers/theclassifier_' + str(number_of_clusters) + '.pickle', 'wb') as f:
        pickle.dump(theclassifier, f, pickle.HIGHEST_PROTOCOL)
