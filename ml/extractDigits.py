import sys, os, logging
import cv2
import imutils
from imutils import contours


def findDigits(filename):

    print("Read image   :   " + filename)
    image = cv2.imread(filename)
    image = imutils.resize(image, height=500, width=500)

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

    cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    
    # Clean up path for imwrite
    filename = filename.split('.')

    i = 0
    for c in cnts:
        
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 5 and (h >= 35 and h <= 80):
            digit = thresh[y-5:y+h+5, x-5:x+w+5]
            digit = cv2.resize(digit,(50,50))

            # Classify img [0-9]
            cv2.imshow('Digit',digit)
            key = cv2.waitKey(0)

            # If a bad picture to classify is shown, press s (skip) to discard img
            if key == ord('s'):
                print("Skipping")
                continue

            savePath = os.path.join('../dataSet', str(chr(key)))

            # If directory does not exist, create new
            if not os.path.isdir(savePath):
                os.mkdir(savePath)
            
            filePath = os.path.join(savePath, filename[0] + str(i) + '.png')
            print("Saving to    :   " + os.path.abspath(filePath))

            # Store img in correct folder for training (50x50) grayscale
            cv2.imwrite(filePath, digit)
        i = i + 1

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    try:
        path = str(sys.argv[1])
        os.chdir(path)
        for file in os.listdir(os.curdir):
            filename = os.fsdecode(file)
            findDigits(filename)
    except IndexError:
        print ("Specify working directory")
        sys.exit(1)

if __name__ == "__main__":
    main()