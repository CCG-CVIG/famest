# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
import cv2


# ------------------------
#   SHAPE DETECTOR
# ------------------------
class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        """
            Detect shape based on the contours of the image
            :param c: list of contours
            :return: shape name
        """
        # Initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)

        # Calculate the shape based on the number of vertices
        # If the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # If the shape has 4 vertices, it is either a square or a rectangle
        elif len(approx) == 4:
            # Compute the bounding box of the contour and use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # A square will have an aspect ratio that is approximately equal to one, otherwise the shape is rectangle
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        # If the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # Otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # Return the name of the shape
        return shape
