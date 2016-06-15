import re
import glob
from collections import Counter

import cv2
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

number_of_clusters = 30

print('run on test script started')

surf = cv2.SURF(1000)

classnames = []
[classnames.append("c"+str(i)) for i in range(10)]

with open("../classifiers/feature_classifier_" + str(number_of_clusters) + ".pickle", "r") as f:
        feature_classifier = pickle.load(f)

print('feature classifier loaded')

with open("../classifiers/theclassifier_"+str(number_of_clusters)+".pickle", 'r') as f:
        theclassifier = pickle.load(f)

print('theclassifier loaded')

image_names = []
for j,img in enumerate(glob.glob("../../statefarm/test/*.jpg")):

    imagename = re.findall("../../statefarm/test/(.*).jpg",img)[0]
    image_names.append(imagename)

    print(imagename)

    image = cv2.imread(img,0)
    kp, des = surf.detectAndCompute(image,None)

    if(des==None):
         interDF = pd.DataFrame(data=np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]), index=None, columns=classnames)
    
    else:
         clus_array = feature_classifier.predict(des)

         cntr =  Counter(clus_array)
         cluster_vector = [cntr[i] for i in range(0,number_of_clusters)]

         cluster_vector = np.array(cluster_vector).reshape(1,(len(cluster_vector)))
    
         label_probs = theclassifier.predict_proba(cluster_vector)
    
         interDF = pd.DataFrame(data=label_probs, index=None, columns=classnames)
    
    if(j==0):
        theresult = interDF
    else:
        theresult = pd.concat([theresult, interDF])

#    theresult.to_csv('inter_submission.csv', index=None, header=True)    

theresult.insert(0, 'img', image_names)

theresult.to_csv('../mebasar_submission.csv', index=None, header=True)

print('results are written to file')
