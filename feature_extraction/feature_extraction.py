import re
import glob

import cv2
import numpy as np
import pandas as pd

print("Feature extraction script started")

surf = cv2.SURF(1000)

print("Surf loaded")

for i in range(10):
    
    for j,img in enumerate(glob.glob("../../statefarm/train/c"+ str(i) +"/*.jpg")):

        imagename = re.findall("../../statefarm/train/c"+ str(i) +"/(.*).jpg",img)[0]
#        print('c'+str(i), j, imagename)
        
        image = cv2.imread(img,0)
        kp, des = surf.detectAndCompute(image,None)
        
        desDF = pd.DataFrame(data=des, index=None, columns=None)
        imglist = np.full((des.shape[0],1),imagename,dtype="S"+str(len(imagename)))
        classnames = np.full((des.shape[0],1),'c'+ str(i),dtype="S2")
        desDF["img"] = imglist
        desDF["classname"] = classnames
        
        if(j==0):
            classfea = desDF
#        elif(j==20):
#            break
        else:
            classfea = pd.concat([classfea, desDF])

    print("c"+ str(i) + "is completed")
    classfea.to_csv("../features/features_c"+str(i)+".csv", index=None)

    print('c'+str(i)+' shape: ', classfea.shape)

    if(i==0):
        result = classfea
    else:
        result = pd.concat([result, classfea])

print("result shape: ",result.shape)

result.to_csv("../features/opencv_features.csv", index=None)

print("features are extracted and written to a file")

