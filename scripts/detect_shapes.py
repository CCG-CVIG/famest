# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from source.shape_detector import ShapeDetector
from matplotlib import pyplot as plt
import numpy as np
import imutils
import cv2
import glob

# Load the image path
# imgs_path = 'photos\\photos_src\\*.jpg'
imgs_path = 'photos\\photos_test\\test03\\*.jpg'

# Check each one of the images
for i, filename in enumerate(glob.iglob(imgs_path)):
    name_file = filename.split('\\')[-1].split('.')[0]
    print('Reading image: {}.jpg'.format(name_file))
    # Load the image and resize it to a smaller factor so that the shapes can be approximated better
    image = cv2.imread(filename)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    # Convert the resized image to a grayscale, blur it slightly and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 0)
    # Find the contours in the image according to the threshold and initialize the shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    mask = np.zeros(blurred.shape, np.uint8)
    cv2.drawContours(mask, cnts, -1, (255, 255, 255), 1)
    # Show the mask
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 600, 600)
    cv2.imshow("Mask", mask)
    sd = ShapeDetector()
    # Loop over the contours
    for c in cnts:
        print("Contour:", c)
        # Compute the center of the contour, then detect the name of the shape using only the contour
        M = cv2.moments(c)
        print("Moments: ", M)
        if 20000 < M["m00"] < 200000:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = sd.detect(c)
            # Multiply the contour (x,y) coordinates by the resize ratio, then draw the contour
            # and name the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print("Shape: ", shape)
            # Show the output image
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", 600, 600)
            cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # Get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            cropped = image[y:y+h, x:x+w]
            # Show the cropped image
            cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cropped", 600, 600)
            cv2.imshow("Cropped", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
