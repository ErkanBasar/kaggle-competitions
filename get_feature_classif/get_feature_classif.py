import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

number_of_clusters = 30

clustered_features = pd.read_csv('../features/clustered_features_' + str(number_of_clusters) + '.csv', sep=',')

feacols = [col for col in clustered_features.columns if col not in ['img','classname','cluster_labels']]

feature_classifier = LogisticRegression(multi_class='ovr', max_iter=1000, n_jobs=-2)

print('Started to classify')

feature_classifier.fit(clustered_features[feacols].values, clustered_features.cluster_labels)

print('Feature classifier is created')

with open("../classifiers/feature_classifier_" + str(number_of_clusters) + ".pickle", 'wb') as f:
        pickle.dump(feature_classifier, f, pickle.HIGHEST_PROTOCOL)

print('Feature classifier written to file')


