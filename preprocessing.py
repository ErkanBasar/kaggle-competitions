import os
import pandas as pd
import numpy as np
import pickle


############# Train Data ################

trn_photo_biz = pd.read_csv("train_photo_to_biz_ids.csv", sep=',')

trn_biz_labels = pd.read_csv("train2.csv", sep=',')

trn_photo_obj = pd.read_csv("train_classified.csv", sep=',')

trn_photo_obj2 = pd.DataFrame()

for i,row in trn_photo_obj.iterrows():
    interlist = [0]*1000
    photo_id = os.path.splitext(row['photo_id'])[0]
    interlist[row['label1']-1] = row['label1_accuracy']
    interlist[row['label2']-1] = row['label2_accuracy']
    interlist[row['label3']-1] = row['label3_accuracy']
    interlist[row['label4']-1] = row['label4_accuracy']
    interlist[row['label5']-1] = row['label5_accuracy']
    trn_photo_obj2 = pd.concat([trn_photo_obj2, pd.Series({photo_id:interlist})])

trn_photo_obj2 = trn_photo_obj2.reset_index()
trn_photo_obj2.columns = ["photo_id","features"]
trn_photo_obj2["photo_id"] = trn_photo_obj2.photo_id.astype('int')

trn_photo_obj2.to_csv("train_photo_to_obj.csv", index=None)

trn_biz_labels2 = trn_biz_labels[~trn_biz_labels['labels'].isnull()] # get rid of the nulls
trn_biz_labels2['labels'] = trn_biz_labels2['labels'].apply(lambda x: [int(y) for y in x.split()])

trn_photo_labels = pd.merge(trn_biz_labels2,trn_photo_biz)

trn_photo_labels.to_csv("train_photo_to_labels.csv", index=None)

trn = pd.merge(trn_photo_labels,trn_photo_obj2, on='photo_id')

trn.to_csv("train_all.csv", index=None)


########### Test Data ##########


test_photo_biz = pd.read_csv("test_photo_to_biz.csv", sep=',')

test_photo_obj = pd.read_csv("test_classified.csv", sep=',')

interlist = np.array([0]*1000)

test_photo_obj2 = pd.DataFrame()

for i,row in test_photo_obj.iterrows():
    interlist = [0]*1000
    photo_id = os.path.splitext(row['photo_id'])[0]
    interlist[row['label1']-1] = row['label1_accuracy']
    interlist[row['label2']-1] = row['label2_accuracy']
    interlist[row['label3']-1] = row['label3_accuracy']
    interlist[row['label4']-1] = row['label4_accuracy']
    interlist[row['label5']-1] = row['label5_accuracy']
    test_photo_obj2 = pd.concat([test_photo_obj2, pd.Series({photo_id:interlist})])


test_photo_obj2 = test_photo_obj2.reset_index()
test_photo_obj2.columns = ["photo_id","features"]
test_photo_obj2["photo_id"] = test_photo_obj2.photo_id.astype('int')

test_photo_obj2.to_csv("test_photo_to_obj.csv", index=None)

test = pd.merge(test_photo_biz, test_photo_obj2, on='photo_id')

test.to_csv("test_all.csv", index=None)

