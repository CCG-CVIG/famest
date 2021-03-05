# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from matplotlib.patches import Rectangle
from matplotlib import pyplot
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from source.metric_extraction import extract_metrics
import numpy as np
import tensorflow as tf
import time
import cv2

# --------------------------------
#  FUNCTIONS
# --------------------------------
def fill_holes(img_th):
    """
        Fill holes in order to remove them from the foot mask
        :param img_th: foot mask
        :return: flood filled image
    """
    # Copy the thresholded image.
    img_floodfill = img_th.copy()
    # Mask used for flood filling
    h, w = img_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    # Combine the two images to get the foreground.
    img_out = img_th | img_floodfill_inv
    return img_out


# --------------------------------
#  PREDICTION CONFIGURATION CLASS
# --------------------------------
class PredictionConfig(Config):
    """
        Defines the prediction configuration
    """
    # Define the name of the configuration
    NAME = "foot_cfg"
    # Number of classes (background + foot)
    NUM_CLASSES = 1 + 1
    # Simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# --------------------------------
#  OBJECT DETECTOR CLASS
# --------------------------------
class ObjectDetector:
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.model_path = model_path
        self.model = MaskRCNN(mode="inference", model_dir="./", config=cfg)
        
# --------------------------------
#  FOOT OBJECT DETECTOR CLASS
# --------------------------------
class FootDetector(ObjectDetector):
    def __init__(self, foot_config, path_to_model):
        super().__init__(cfg=foot_config, model_path=path_to_model)
        # Load model weights
        self.model.load_weights(self.model_path, by_name=True)
        # Get default session
        self.graph = tf.get_default_graph()

    def plot_image(self, image):
        with self.graph.as_default():
            # Make prediction
            result = self.model.detect([image], verbose=1)[0]
            mask = result["masks"]
            mask = mask.astype(int)
            # Print detection time
            # detection_time = time.time() - load_time
            # print("Detection Time: {} ms".format(detection_time))
            # Show image with the detected feet
            #fig = pyplot.figure(figsize=(8, 8))
            #fig.add_subplot(1, 3, 1)
            #pyplot.imshow(image)
            ax = pyplot.gca()
            for box in result["rois"]:
                # get coordinates
                y1, x1, y2, x2 = box
                # calculate width and height of the box
                width, height = x2 - x1, y2 - y1
                # create the shape
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                # draw the box
                ax.add_patch(rect)
            okay_flag = False
            res = None
            for i in range(mask.shape[2]):
                #print("Score: ", result["scores"][i])
                if result["scores"][i] >= 0.99:
                    temp = image.copy()
                    for j in range(temp.shape[2]):
                        temp[:, :, j] = temp[:, :, j] * mask[:, :, i]
                    res = apply_watershed(temp, mask[:, :, i])
                    #fig.add_subplot(1, 3, i + 2)
                    #pyplot.imshow(mask[:, :, i])
                    okay_flag = True
                    break
            #pyplot.show()
            return res, okay_flag

    def metric_extract(self, img, input_type):
        paper_points, extract_img = extract_metrics(img, input_type)
        foot_segmented, okay_flag = self.plot_image(extract_img)
        if foot_segmented is not None:
            #print(foot_segmented)
            foot_segmented = fill_holes(np.uint8(foot_segmented))
            #show_image(foot_segmented, "FOOT")
        else:
            okay_flag = False
        return paper_points, foot_segmented, okay_flag






# ------------------------------
#   FUNCTIONS
# ------------------------------
def apply_watershed(image, mask):
    """
        Applies the watershed algorithm to the image according to a given mask
        :param image: input image
        :param mask: input mask
        :return: watershed image
    """
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


# def init_model(input_type):
#     """
#         Initialize the Mask RCNN model
#         :param input_type: Type of input file
#         :return:
#     """
#     # Create config
#     cfg = PredictionConfig()
#     # Define the model
#     model = MaskRCNN(mode="inference", model_dir="./", config=cfg)
#     model_path = ""
#     # Load model weights
#     if input_type == "photo":
#         model_path = "data/mask_rcnn_foot_cfg_0050_5.h5"
#     else:
#         model_path = "data/mask_rcnn_foot_cfg_0050_5.h5"
#     model.load_weights(model_path, by_name=True)
#     return model

# def init_model():
#     """
#         Initialize the Mask RCNN model
#     """
#     # Create config
#     cfg = PredictionConfig()
#     # Define the model
#     model = MaskRCNN(mode="inference", model_dir="./", config=cfg)
#     model_path = "data/mask_rcnn_foot_cfg_0050_5.h5"
#     # Load model weights
#     model.load_weights(model_path, by_name=True)
#     return model


# def plot_image(image, model):
#     """
#         Plot the image with the detections
#         :param image: input image
#         :param model: input data model weights
#         :return:
#     """
#     # Start counting time
#     # start_time = time.time()
#     # Get default graph in TF session
#     # graph = tf.get_default_graph()
#     # with graph.as_default():

#     # Make prediction
#     result = model.detect([image], verbose=1)[0]
#     mask = result["masks"]
#     mask = mask.astype(int)
#     # Print detection time
#     # detection_time = time.time() - load_time
#     # print("Detection Time: {} ms".format(detection_time))
#     # Show image with the detected feet
#     #fig = pyplot.figure(figsize=(8, 8))
#     #fig.add_subplot(1, 3, 1)
#     #pyplot.imshow(image)
#     ax = pyplot.gca()
#     for box in result["rois"]:
#         # get coordinates
#         y1, x1, y2, x2 = box
#         # calculate width and height of the box
#         width, height = x2 - x1, y2 - y1
#         # create the shape
#         rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#         # draw the box
#         ax.add_patch(rect)
#     okay_flag = False
#     res = None
#     for i in range(mask.shape[2]):
#         #print("Score: ", result["scores"][i])
#         if result["scores"][i] >= 0.99:
#             temp = image.copy()
#             for j in range(temp.shape[2]):
#                 temp[:, :, j] = temp[:, :, j] * mask[:, :, i]
#             res = apply_watershed(temp, mask[:, :, i])
#             #fig.add_subplot(1, 3, i + 2)
#             #pyplot.imshow(mask[:, :, i])
#             okay_flag = True
#             break
#     #pyplot.show()
#     return res, okay_flag
