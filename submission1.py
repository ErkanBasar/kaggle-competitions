import os
import pandas as pd
import numpy as np
import pickle
import ast
from scipy import sparse
import logging

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import  ElasticNetCV
from sklearn.svm import SVC

logging.basicConfig(
		format='%(asctime)s, %(levelname)s: %(message)s',
		filename='data/doalls.log',
		datefmt='%d-%m-%Y, %H:%M',
		level=logging.INFO)

logging.info('initialized')


########## Train ############

trn = pd.read_csv("train_all.csv", sep=',')

labels_real = []
for l in trn['labels'].values:
    labels_real.append(ast.literal_eval(l))

la = MultiLabelBinarizer()
labelsarray = la.fit_transform(labels_real)

logging.info('labels are ready')

features = np.array(trn.features.tolist())

features_real = []
for l in features:
    features_real.append(ast.literal_eval(l))

logging.info('train features ready')

fea = sparse.lil_matrix(features_real)

logging.info('train features are sparse')

classif = OneVsRestClassifier(ElasticNetCV(max_iter=10000), n_jobs=-2)

logging.info('started to classify')

classif.fit(fea,labelsarray)

with open('classifier_doalls.pickle', 'wb') as f:
        pickle.dump(classif, f, pickle.HIGHEST_PROTOCOL)

logging.info('classified')

######### Test ##########

test = pd.read_csv("test_all.csv", sep=',')

test_features = np.array(test.features.tolist())

test_features_real = []
for l in test_features:
    test_features_real.append(ast.literal_eval(l))

logging.info('test features ready')

sparse_test_fea = sparse.lil_matrix(test_features_real)

logging.info('test features sparse')

labelarray = classif.predict(sparse_test_fea)

logging.info('prediction complete')

labels = la.inverse_transform(labelarray)

logging.info('labels length :', len(labels))

test['labels'] = labels

testDF = test.drop('features', 1)

test_final = testDF.drop('photo_id', 1)

for i,t in enumerate(test_final['labels'].values):
    test_final['labels'].values[i] = ' '.join(map(str,t))

test_final.to_csv("test_final.csv", index=None)

logging.info('important part is done')

with open('labels.txt', 'wb') as f:
	f.write(labels)

logging.info('all done')