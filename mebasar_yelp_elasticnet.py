import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import  ElasticNetCV

trn = pd.read_csv('train.csv', sep=',', header=None)

la = MultiLabelBinarizer()
la.fit_transform([[0,1,2,3,4,5,6,7,8],[2,4,6,7,8]])


features = []
labels = []
test_feats = []


for r in range(len(trn)):
    tmpfeat = []
    for c in range(1,246):
        tmpfeat.append(trn[c][r])
    features.append(tmpfeat)

trn['features'] = features

for r in range(len(trn)):
    tmplabels = []
    for c in range(246,255):
        tmplabels.append(trn[c][r])
    labels.append(tmplabels)

trn['labels'] = labels


classif = OneVsRestClassifier(ElasticNetCV(max_iter=10000), n_jobs=-2)

np_labels = np.array(labels)


test = pd.read_csv('test.csv', sep=',', header=None)


for r in range(len(test)):
    tmpfeat = []
    for c in range(1,246):
        tmpfeat.append(test[c][r])
    test_feats.append(tmpfeat)


test['features'] = test_feats


labelarray = classif.predict(test_feats)


labels = la.inverse_transform(labelarray)


sub = pd.DataFrame({"business_id":test[0].values, "labels":test['labels'].values})

for i,t in enumerate(sub['labels'].values): 
    sub['labels'].values[i] = ' '.join(map(str,t))


sub.to_csv('mebasar_yelp_submission.csv', index=None)