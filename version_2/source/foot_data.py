import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

from matplotlib.patches import Rectangle
from matplotlib import pyplot
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from numpy import mean

import numpy as np
import time
import cv2

"""
Define the prediction configuration
"""
class PredictionConfig(Config):

	# Define the name of the configuration
	NAME = "foot_cfg"

	# Number of classes (background + foot)
	NUM_CLASSES = 1 + 1

	# Simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

################################################################################
### Apply watershed algorithm to image
################################################################################
def apply_watershed(image, mask):

	# Convert BGR to HSV
	ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	y_img = ycrcb[:, :, 0]
	res = np.ones(y_img.shape) * 255
	changed_pixels = np.zeros(y_img.shape)

	mask_y_img = y_img[mask > 0]
	mean_y = np.mean(mask_y_img)

	res[y_img > mean_y] = 0
	res[mask == 0] = 0
	return res

def init_model():

	# Create config
	cfg = PredictionConfig()

	# Define the model
	model = MaskRCNN(mode = "inference", model_dir = "./", config = cfg)

	# Load model weights
	model_path = "models/mask_rcnn_foot_cfg_0050.h5"
	model.load_weights(model_path, by_name = True)

	return model

################################################################################
### Plot image with detections
################################################################################
def plot_image(image, model):

	# Start counting time
	#start_time = time.time()

	# Make prediction
	result = model.detect([image], verbose = 1)[0]
	mask = result["masks"]
	mask = mask.astype(int)

	# Print detection time
	# detection_time = time.time() - load_time
	# print("Detection Time: {} ms".format(detection_time))

	# Show image with the detected feet
	fig = pyplot.figure(figsize = (8, 8))
	fig.add_subplot(1, 3, 1)
	pyplot.imshow(image)
	ax = pyplot.gca()
	for box in result["rois"]:
		# get coordinates
		y1, x1, y2, x2 = box
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')
		# draw the box
		ax.add_patch(rect)

	okay_flag = False
	res = None
	for i in range(mask.shape[2]):

		if result["scores"][i] >= 0.99:

			temp = image.copy()
			for j in range(temp.shape[2]):
				temp[:, :, j] = temp[:, :, j] * mask[:, :, i]

			res = apply_watershed(temp, mask[:, :, i])
			fig.add_subplot(1, 3, i + 2)
			pyplot.imshow(mask[:, :, i])

			okay_flag = True
			break

	pyplot.show()
	return res, okay_flag
