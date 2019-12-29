import numpy as np
import os
import scipy.ndimage
import imageio
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

features_list = []
features_label = []

# load labeled training / test data
# loop over the 10 directories where each directory stores the images of a digit
for digit in range(0,10):
    label = digit
    training_directory = 'dataset/' + str(label) + '/'
    for filename in os.listdir(training_directory):
        if (filename.endswith('.png')):
            training_digit = imageio.imread(training_directory + filename)
            training_digit = color.rgb2gray(training_digit)

            # extra digit's Histogram of Gradients (HOG). Divide the image into 5x5 blocks and where block in 10x10 pixels
            df= hog(training_digit, orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
       
            features_list.append(df)
            features_label.append(label)

# store features array into a numpy array
features  = np.array(features_list, 'float64')

# split the labled dataset into training / test sets
X_train, X_test, y_train, y_test = train_test_split(features, features_label)

# train using K-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# get the model accuracy
model_score = knn.score(X_test, y_test)
print(model_score)

# save trained model
joblib.dump(knn, '../model/knn_model.pkl')

