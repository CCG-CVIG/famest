# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import base64
import datetime
import io
import json
import os
import shutil
import sys
import time
import traceback
import uuid
import tqdm
import numpy as np
import requests
import glob
import cv2
import waitress
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
from zipfile import ZipFile
from flask import *
from flask_cors import CORS
from flask_restplus import Api, Resource, fields, reqparse
from source.metric_extraction import extract_metrics
from source.foot_reconstruction import get_focal_length, cut_visual_hull, cloud, surface_cloud, parameters
from source.foot_data import init_model

# ------------------------
#   GLOBAL VARIABLES
# ------------------------
authorizations = {
    'Basic Authentication': {
        'type': 'basic',
        'in': 'header',
        'name': 'Authorization'
    },
}
apiVersion = 'API Version 3.0'
flask_app = Flask(__name__)
flask_app.url_map.strict_slashes = False
CORS(flask_app)
app = Api(app=flask_app, version="3.0", title="FAMEST Platform", doc="/swagger/",
          description="FAMEST Platform Point Cloud Processing Routines.")
# START = time.time()
X = np.load("models/hull.npy")
X = X[0, :, :]
TOTAL_POINTS = len(X)
K = np.float64([[3666, 0, 2048], [0, 3666, 1548], [0, 0, 1]])   # Just for init

# ------------------------
#   NAMESPACES
# ------------------------
debug_ns = app.namespace('debug', descriptions="Debug operations")
api_ns = app.namespace('api', description="Image API operations")


# ------------------------
#   FUNCTIONS
# ------------------------
def show_image(img, name="Image", time=0):
    """
        Show the image in an OpenCV window
        :param img: Input Image
        :param name: Name of the Window
        :param time: Time for waitKey
        :return:
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def show_image_file(filename, name="Image File", time=0):
    """
        Show the image file in an OpenCV window
        :param filename: Input file
        :param name: Name of the OpenCV window
        :param time: Time for waitKey
        :return:
    """
    if ".jpg" in filename:
        # Create window
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
        # Read image from image file in BGR format
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Show image content
        cv2.imshow(name, img)
        cv2.waitKey(time)
        # Close all the frames
        cv2.destroyAllWindows()
    elif ".mp4" in filename:
        cap = cv2.VideoCapture(filename)
        # Check if camera opened successfully
        if not cap.isOpened():
            #print("Error opening video stream or file!")
            return
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Display the resulting frame
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.imshow(name, frame)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break from loop
            else:
                break
        # When everything is done, release the video capture object
        cap.release()
        # Close all the frames
        cv2.destroyAllWindows()


def zip_folder(path, folderName, return_path):
    zip_path = '{}/{}.zip'.format(path, str(uuid.uuid4()))
    with ZipFile(zip_path, 'w') as zipObj:
        # Iterate over all the files in directory
        os.chdir(folderName)
        for folderName, _, filenames in os.walk('./'):
            for filename in filenames:
                # Add file to zip
                zipObj.write(filename)
        os.chdir(return_path)
    return os.path.abspath(zip_path)


def unzip_folder(filePathName, filePathFolder):
    # Unzip files in file path folder
    with ZipFile(filePathName, 'r') as zip_ref:
        zip_ref.extractall(filePathFolder)


# ------------------------
#  API CLASSES/ENDPOINTS
# ------------------------
@api_ns.route("/version")
class Version(Resource):
    @api_ns.doc(responses={200: 'OK', 400: 'Bad Request'})
    def get(self):
        try:
            return {
                'status': 'OK',
                'apiVersion': apiVersion
            }
        except Exception as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(400, error.__doc__, status="Bad Request!", statusCode="400")


@api_ns.route("/log")
class Log(Resource):
    @api_ns.doc(responses={200: 'OK', 400: 'Bad Request'})
    @api_ns.produces(['text/plain'])
    def get(self):
        try:
            with open('/code/err.log') as f:
                log = f.read()
            resp = Response(log)
            resp.headers['Content-type'] = 'text/plain'
            return resp
        except Exception as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(400, error.__doc__, status="Bad Request!", statusCode="400")


@api_ns.route("/metrics")
class PlyMetrics(Resource):
    @api_ns.doc(responses={200: 'OK', 400: 'Bad Request!'})
    def get(self):
        try:
            return send_file('sheet_dimensions.txt')
        except requests.exceptions.HTTPError as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(error.response.status_code, error.__doc__, status="Bad Request!",
                      statusCode="{}".format(error.response.status_code))
        except Exception as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(400, error.__doc__, status="Bad Request!", statusCode="400")

#@api_ns.route("/metrics", methods=['POST'])
#class PlyMetrics(Resource):
#    @api_ns.doc(responses={200: 'OK', 400: 'Bad Request!'})
#    def post(self):
#        try:
#            return send_file('sheet_dimensions.txt')
#        except requests.exceptions.HTTPError as error:
#            with open("/code/err.log", "a+") as f:
#                f.write(str(traceback.format_exc()) + "\n")
#            app.abort(error.response.status_code, error.__doc__, status="Bad Request!",
#                      statusCode="{}".format(error.response.status_code))
#        except Exception as error:
#            with open("/code/err.log", "a+") as f:
#                f.write(str(traceback.format_exc()) + "\n")
#            app.abort(400, error.__doc__, status="Bad Request!", statusCode="400")


upload_parser = app.parser()
upload_parser.add_argument('file_data', location='files',
                           type=FileStorage, required=True)


@api_ns.route("/process", methods=['POST'])
@api_ns.expect(upload_parser)
class StartProcessing(Resource):
    @api_ns.doc(responses={200: 'OK', 400: 'Bad Request'})
    def post(self):
        global X
        try:
            # Get files from client side
            args = upload_parser.parse_args()
            upload_file = args["file_data"]
            filename = upload_file.filename
            filePathName = None
            name = None
            # Check file info to create directory
            if ".zip" in filename:
                name, extension = filename.split(".zip")
                # Create file directory
                directory = "photos/FAMEST/{}".format(name)
                os.mkdir(directory)
                # Create file path name
                filePathName = os.path.join(directory, filename)
            elif ".jpg" in filename:
                name, extension = filename.split(".jpg")
                # Create file directory
                directory = "photos/FAMEST/{}".format(name)
                os.mkdir(directory)
                # Create file path name
                filePathName = os.path.join(directory, filename)
            elif ".mp4" in filename:
                name, extension = filename.split(".mp4")
                # Create file directory
                directory = "videos/FAMEST/{}".format(name)
                os.mkdir(directory)
                # Create file path name
                filePathName = os.path.join(directory, filename)
            # Write the data to the file path
            upload_file.save(filePathName)
            # Wait for a bit
            time.sleep(5)
            start = time.time()
            if ".zip" in filename:
                # Load image model
                model = init_model("photo")
                # Unzip .zip file and place files in folder
                unzip_folder(filePathName, "photos/FAMEST/{}".format(name))
                # Look at unzip folder
                path = "photos/FAMEST/{}/*.jpg".format(name)
                filenames = [img for img in glob.iglob(path)]
                filenames.sort()
                for i, filename in enumerate(filenames):
                    # Metric extraction
                    name_file = filename.split('/')[-1].split('.')[0]
                    #print('Reading image: {}.jpg'.format(name_file))
                    save_folder = '/'.join(filename.split('/')[:-1]) + '/'
                    numpy_filename = save_folder + name_file + '.npy'
                    image = cv2.imread(filename)
                    paper_points, foot_segmented, okay_flag = extract_metrics(image, model, "photo")
                    # 3D reconstruction
                    if okay_flag:
                        paper_points = np.float64(paper_points)
                        X = cut_visual_hull(image, foot_segmented, K, paper_points, X)
                        #print("Hull size of '{}': {}".format(name_file, len(X)))
            elif ".jpg" in filename:
                # Load image model
                model = init_model("photo")
                # Look at images folder
                path = "photos/FAMEST/{}/*.jpg".format(name)
                filenames = [img for img in glob.iglob(path)]
                filenames.sort()
                for i, filename in enumerate(filenames):
                    # Metric extraction
                    name_file = filename.split('/')[-1].split('.')[0]
                    #print('Reading image: {}.jpg'.format(name_file))
                    save_folder = '/'.join(filename.split('/')[:-1]) + '/'
                    numpy_filename = save_folder + name_file + '.npy'
                    image = cv2.imread(filename)
                    paper_points, foot_segmented, okay_flag = extract_metrics(image, model, "photo")
                    # 3D reconstruction
                    if okay_flag:
                        paper_points = np.float64(paper_points)
                        X = cut_visual_hull(image, foot_segmented, K, paper_points, X)
                        #print("Hull size of '{}': {}".format(name_file, len(X)))
            elif ".mp4" in filename:
                # Load video model
                model = init_model("video")
                # Look at video folder
                path = filePathName
                cap = cv2.VideoCapture(path)
                i = 0
                numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened():
                    #print("Frame {}/{}".format(i, numFrames))
                    _, image = cap.read()
                    if i % 30 == 0:
                        if image is not None:
                            paper_points, foot_segmented, okay_flag = extract_metrics(image, model, "video")
                        else:
                            okay_flag = False
                        # 3D reconstruction
                        if okay_flag:
                            paper_points = np.float64(paper_points)
                            X = cut_visual_hull(image, foot_segmented, K, paper_points, X)
                            #print("Hull size of '{}': {}".format(filename, len(X)))
                    i += 1
                    if i == numFrames:
                        # Show image
                        #if image is not None:
                        #    show_image(image)
                        break
                # When everything is set and done release the video capture object
                cap.release()
            # Create foot point cloud
            obj = np.float64(
                [[-29 / 2, -21 / 2, 0], [29 / 2, -21 / 2, 0], [-29 / 2, 21 / 2, 0], [29 / 2, 21 / 2, 0]])
            X[:, 2] = -X[:, 2]
            # Save cloud files in point cloud directory
            directory = "models/{}".format(name)
            os.mkdir(directory)
            np.save('{}/foot_bulk.npy'.format(directory), X)
            cloud('{}/foot_bulk.ply'.format(name), np.transpose(X), paper=False)
            # Create surface point cloud
            Z = surface_cloud(X, name)
            cloud('{}/surface_cloud.ply'.format(name), Z, paper=False)
            parameters(name)
            # Close timer
            end = time.time()
            #print('Final time: {0:.2f}'.format(end - start))
            # Send .ply file
            # with open('{}\\surface_cloud.ply'.format(directory), "r") as file:
            return send_file('{}/surface_cloud.ply'.format(directory))
        except requests.exceptions.HTTPError as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(error.response.status_code, error.__doc__, status="Bad Request!",
                      statusCode="{}".format(error.response.status_code))
        except Exception as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(400, error.__doc__, status="Bad Request!", statusCode="400")


@api_ns.route("/ply/trigger/")
class PlyTrigger(Resource):
    @api_ns.doc(responses={200: 'OK', 400: 'Bad Request'})
    def post(self):
        try:
            # Get images from client side
            file_token = request.headers['Authorization']
            file_id = request.form.get('fileId')
            with open("/code/err.log", "a+") as f:
                f.write(str(file_id) + "\n")
            payload = {
                'tenantId': request.form.get('tenantId'),
                'assetId': request.form.get('assetId'),
                'deviceId': request.form.get('deviceId'),
                'namespace': request.form.get('namespace'),
                'topic': request.form.get('topic'),
                'subscription': request.form.get('subscription'),
                'fileName': request.form.get('fileName')
            }
            # TODO
            return {
                'status': 'OK'
            }
        except requests.exceptions.HTTPError as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(error.response.status_code, error.__doc__, status="Bad Request!",
                      statusCode="{}".format(error.response.status_code))
        except Exception as error:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(400, error.__doc__, status="Bad Request!", statusCode="400")


# ------------------------
# DEBUG CLASSES/ENDPOINTS
# ------------------------
@debug_ns.route("/image/readimages/<string:project>/<string:mode>")
class ListImages(Resource):
    @debug_ns.doc(responses={200: 'OK', 400: 'Bad Request'},
                  params={'project': 'Cube ID.', 'mode': 'imgs | prep | proc'})
    def get(self, project, mode):
        try:
            # TODO

            return {
                'status': 'OK',
                'filenames': 'originals'
            }
        except Exception as err:
            with open("/code/err.log", "a+") as f:
                f.write(str(traceback.format_exc()) + "\n")
            app.abort(400, err.__doc__, status="Bad Request!", statusCode="400")


# ------------------------
#   MAIN
# ------------------------
if __name__ == '__main__':
    print(apiVersion)
    waitress.serve(flask_app, host="0.0.0.0", port=19100)
