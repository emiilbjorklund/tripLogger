import numpy as np
import os
import scipy.ndimage
import imageio
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

knn = joblib.load('model/knn_model.pkl')
image = imageio.imread('dataSet/9/IMG_49421.png')


image = color.rgb2gray(image)
df= hog(image, orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))

predict = knn.predict(df.reshape(1,-1))[0]
print(predict)


