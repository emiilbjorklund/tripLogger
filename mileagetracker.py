import os
import sys
import csv
import cv2
import numpy as np
import imutils
from imutils import contours
from exif import Image
from geopy.geocoders import Nominatim
from sklearn.externals import joblib
from skimage.feature import hog

knn = joblib.load('model/knn_model.pkl')

logEntry = {
    'Date'          : None,
    'MileageStart'  : None,
    'MileageEnd'    : None,
    'Length'        : None,
    'TripStart'     : None,
    'Errand'        : None,
    'TripEnd'       : None,
}

class ImageData:
    def __init__(self,filepath):
        self.path = filepath
        self.cvImg = cv2.imread(self.path)
        self.date = ''
        self.cord = ''
        self.digits = []

        self.getMetaData()
        self.processImage()

    def getMetaData(self):
        with open(self.path, 'rb') as image_file:
            image = Image(image_file)
        lat = str(image.gps_latitude[0] + image.gps_latitude[1]/60 + image.gps_latitude[2]/3600)
        log = str(image.gps_longitude[0] + image.gps_longitude[1]/60 + image.gps_longitude[2]/3600)
        self.cord = lat+","+log

        split = image.datetime.split()
        self.date = split[0].replace(':','-')

    def processImage(self):
        image = imutils.resize(self.cvImg, height=500, width=500)
        #Pre-prossesing of img
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        inv =~blurred
        thresh = cv2.threshold(inv, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        #Find the each number
        cnts = cv2.findContours(thresh,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = imutils.grab_contours(cnts)
        cnts = self.sort_contours(cnts,method="left-to-right")[0]

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            
            #if w >= 2 and (h >= 0 and h <= 200):
            digit = thresh[y-5:y+h+5, x-5:x+w+5]
            digit = cv2.resize(digit,(50,50))
            self.digits.append(digit)
        
        

    def sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

def predictNumber(image):
    number = ''
    for digit in image.digits:
        df= hog(digit, orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
        predict = knn.predict(df.reshape(1,-1))[0]
        predict_proba = knn.predict_proba(df.reshape(1,-1))
        #print(predict_proba[0][predict])
        number+=str(predict)
    
    return number

def getLocation(image):
    adress = ''
    geolocator = Nominatim(user_agent="mileageTracker")
    location = geolocator.reverse(image.cord)
    adress = location.raw['address']['postcode'] + ', ' + location.raw['address']['city']
    return str(adress)

def appendLog(key,value):
    logEntry[key] = value

def main():
    images = []

    #os.system('cls' if os.name == 'nt' else 'clear')
    try:
        for path in sys.argv[1:]:
            images.append(ImageData(path))

    except IndexError:
        print ("Specify working directory")
        sys.exit(1)

    appendLog('Date',images[0].date)
    appendLog('MileageStart',predictNumber(images[0]))
    appendLog('MileageEnd',predictNumber(images[1]))
    appendLog('Length',str(int(logEntry['MileageEnd'])-int(logEntry['MileageStart'])))
    appendLog('TripStart',getLocation(images[0]))
    appendLog('TripEnd',getLocation(images[1]))
    appendLog('Errand', 'Test Run')

    with open('log.csv', mode='a') as log_file:
        entry = csv.DictWriter(log_file, logEntry.keys(), delimiter=";")
        entry.writerow(logEntry)

    
    

    

if __name__ == "__main__":
    main()