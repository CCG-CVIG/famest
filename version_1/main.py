import sys
import cv2
import numpy as np
import os
import math
import gc
import time
import glob

from source.metric_extraction import extract_metrics
from source.foot_reconstruction import get_focal_length, cut_visual_hull, cloud, surface_cloud, parameters

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("ERROR: Insert the folder name")
        sys.exit(0)

    type = sys.argv[1]
    name = sys.argv[2]
    start = time.time()

    X = np.load("models\\hull.npy")
    X = X[0, :, :]
    Total_points = len(X)
    K = np.float64([[3666, 0, 2048], [0, 3666, 1548], [0, 0, 1]])

    if type == "photo":

        # Images Folder
        path = "photos\\{}\\*.jpg".format(name)
        for i, filename in enumerate(glob.iglob(path)):
            # Metric extraction
            name_file = filename.split('\\')[-1].split('.')[0]

            save_folder = '\\'.join(filename.split('\\')[:-1]) + '\\'
            numpy_filename = save_folder + name_file + '.npy'
            image = cv2.imread(filename)
            paper_points, foot_segmented, okay_flag = extract_metrics(image)

            # if i == 0:
            #     fx, fy = get_focal_length(image.shape, paper_points)
            #     cx = image.shape[1] / 2
            #     cy = image.shape[0] / 2
            #     K = np.float64([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # 3D reconstruction
            if okay_flag:
                paper_points = np.float64(paper_points)
                X = cut_visual_hull(image, foot_segmented, K, paper_points, X)
                print("Hull size of '{}': {}".format(name, len(X)))
    else:
        path = "photos\\{}".format(name)
        cap = cv2.VideoCapture(path)
        i = 0

        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while (cap.isOpened()):

            _, image = cap.read()
            if i % 30 == 0:
                paper_points, foot_segmented, okay_flag = extract_metrics(image)

                # if i == 0:
                #     fx, fy = get_focal_length(image.shape, paper_points)
                #     cx = image.shape[1] / 2
                #     cy = image.shape[0] / 2
                #     K = np.float64([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                # 3D reconstruction
                if okay_flag:
                    paper_points = np.float64(paper_points)
                    X = cut_visual_hull(image, foot_segmented, K, paper_points, X)
                    print("Hull size: {}".format(len(X)))

            i += 1
            if (i == nframes):
                break

        cap.release()

    obj = np.float64([[-29/2, -21/2, 0], [29/2, -21/2, 0], [-29/2, 21/2, 0], [29/2, 21/2, 0]])
    X[:, 2] = -X[:, 2]
    np.save('models\\foot_bulk.npy', X)
    cloud('foot_bulk.ply', np.transpose(X), paper = False)

    Z = surface_cloud(X)
    cloud('surface_cloud.ply', Z, paper = False)
    parameters()

    end = time.time()
