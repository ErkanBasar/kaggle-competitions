from collections import Counter

import pickle
import numpy as np
import pandas as pd

number_of_clusters = 30

clustered_features = pd.read_csv('../features/clustered_features_' + str(number_of_clusters) + '.csv', sep=',')

names = []
[names.append("cl"+str(i)) for i in range(number_of_clusters)]
names.append('img')
names.append('classname')

listoflists = []
for img in clustered_features.img.unique():

    imgDF = clustered_features.loc[clustered_features.img == img]
    clusterlist = imgDF.cluster_labels.tolist()
    
    cntr =  Counter(clusterlist)
    interlist = [cntr[i] for i in range(0,number_of_clusters)]
        
    interlist.append(img)
    interlist.append(list(imgDF.classname)[0])
    
    listoflists.append(interlist)

image_cl = pd.DataFrame(data=listoflists, index=None, columns=names)

image_cl.to_csv("../features/cluster_vectors_" + str(number_of_clusters) + ".csv", index=None)
