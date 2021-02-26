import imageio
import sys
import os

root = os.getcwd()
src = "{}/Source".format(root)
sys.path.append(src)

from foot_data import plot_image

"""
Main function
"""
if __name__ == "__main__":

	if (len(sys.argv) == 1):
		print("ERROR: invalid number of args")
		sys.exit(0)

	image_name = sys.argv[1]
	image = imageio.imread("Images/{}".format(image_name))

	model_path = "{}/Data/mask_rcnn_foot_cfg_0050.h5".format(root)
	plot_image(image, model_path)
