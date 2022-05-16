import numpy as np 
import cv2 
import os
import pandas as pd
import glob
from skimage.feature import greycomatrix, greycoprops

imgs = []
labels = []
for img in glob.glob("Path/*.jpg"):
            base = os.path.basename(img)
            img = cv2.imread(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           
            resize = cv2.resize(gray,(200,200))
           
            matrix_coocurrence = greycomatrix(resize, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=False, symmetric=False)
            
            imgs.append(resize)
            labels.append(base)

            cv2.imshow("Ukuran Citra Asli", img)
            cv2.imshow("Ukuran Citra Resize",resize)
            cv2.waitKey(0)         

from skimage.feature import greycomatrix, greycoprops


def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature



properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

glcm_all_agls = []
for img, label in zip(imgs, labels): 
    glcm_all_agls.append(
            calc_glcm_all_agls(img, 
                                label, 
                                props=properties)
                            )
columns = []
angles = ['0', '45', '90','135']
for name in properties :
    for ang in angles:
        columns.append(name + "_" + ang)
        
columns.append("label")

import pandas as pd 

glcm_df = pd.DataFrame(glcm_all_agls, 
                      columns = columns)

glcm_df.to_csv("DataTrain.csv")


