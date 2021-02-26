# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
import cv2
import numpy as np
import math
import time
import os
import gc


# ----------------------------------------------
#   FUNCTIONS
# ----------------------------------------------
def get_focal_length(image_shape, paper_points):
    """
        Get camera focal length position according to paper corners
        :param image_shape: input image
        :param paper_points: paper corners
        :return: position coordinates of the camera focal length position
    """
    h, w, _ = image_shape
    sheet_width = 254
    sheet_height = 108
    dist = 450
    # Get the paper center point
    x = [p[1] for p in paper_points]
    y = [p[0] for p in paper_points]
    center = (sum(x) / len(paper_points), sum(y) / len(paper_points))
    fx = center[1] * dist / sheet_width
    fy = center[0] * dist / sheet_height
    return fx, fy


def create_hull():
    """
        Create Hull Function
        :return: Hull mask with 3 million points
    """
    hull = []
    for i in range(0, 273):             # Length
        for j in range(0, 126):         # Width
            for k in range(0, 101):     # Height
                h = [[-14.5 + i/5], [-12.5 + j/5], [0 - k/5]]
                hull.append(h)
                del h
    hull = np.float32(hull).reshape(1, -1, 3)
    np.save("hull.npy", hull)


def get_3D(paper_corners, K):
    """
        Get the 3D coordinates according to the paper corners and the intrinsic parameters from the camera
        :param paper_corners: list of paper corners ( [[1022,1242],[],[],[]] (te,tr,be,br) )
    """
    objectPoints = np.float64([[-29/2, -21/2, 0], [29/2, -21/2, 0], [-29/2, 21/2, 0], [29/2, 21/2, 0]])
    _, rvec, tvec = cv2.solvePnP(objectPoints, paper_corners, K, None)
    return rvec, tvec


def cut_visual_hull(img, foot_segmented, K, points, hull):
    """
        Cut the visual hull from the image according to a given foot mask
        :param img: Input Image
        :param foot_segmented: Foot Mask
        :param K: Intrinsic Camera Parameters
        :param points: List of points
        :param hull: Hull points
        :return: Cut Hull Image
    """
    # Given a specific image, foot mask, camera intrinsics, points of paper and hull, cuts the hull based on the mask
    _img = img.copy()
    rvec, tvec = get_3D(points, K)
    gc.collect()
    # Project the points from the 3D coordinates of the object
    a, _ = cv2.projectPoints(hull, rvec, tvec, K, None)
    a = np.rint(a).astype(int)
    a = a[:, 0, :]
    if len(foot_segmented.shape) > 2:
        img_gray = cv2.cvtColor(foot_segmented, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = foot_segmented
    kernel = np.ones((5, 5), np.uint8)
    img_gray = cv2.dilate(img_gray, kernel, iterations=20)
    img_gray = cv2.erode(img_gray, kernel, iterations=20)
    img_mask = np.zeros_like(img_gray)
    img_mask[img_gray > 0] = 255
    c = np.copy(a)
    a = a[np.where((a[:, 1] < img_gray.shape[0]) & (a[:, 1] >= 0) & (a[:, 0] < img_gray.shape[1]) & (a[:, 0] >= 0))]
    Z = np.copy(hull)
    Z = Z[np.where((c[:, 1] < img_gray.shape[0]) & (c[:, 1] >= 0) & (c[:, 0] < img_gray.shape[1]) & (c[:, 0] >= 0))]
    A = np.zeros(a.shape[0])
    A[img_mask[a[:, 1], a[:, 0]] == 255] = 1
    Z = Z[A == 1]
    gc.collect()
    return Z


def cloud(file_path, vec, paper=False):
    """
        Create a point cloud and save it in a .ply file
        :param file_path: File path to save the .ply file
        :param vec: Vector points
        :param paper: Paper flag
        :return: Point cloud in a .ply file
    """
    vec = np.float32(vec)
    file_path = 'models/' + file_path
    filepath = open(file_path,"w")
    filepath.write('ply\n')
    filepath.write('format ascii 1.0\n')
    filepath.write('element vertex {}\n'.format(vec.shape[1]))
    filepath.write('property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n')
    if paper:
        filepath.write('element face 1\n')
        filepath.write('property list uchar int vertex_index\n')
    filepath.write('end_header\n')
    final = np.transpose(vec[0:3])
    for i in range(0, final.shape[0]):
        for n in range(0, 3):
            filepath.write(' {}'.format(final[i, n]))
        for n in range(3, 6):
            filepath.write(' 0')
        filepath.write('\n')
    if paper:
        filepath.write('4 0 1 3 2')
    filepath.close()


def surface_cloud(Z, directory):
    """
        Create a surface point cloud from the points extracted from the surface of the bulk cloud
        :param: directory: Input path to save surface point cloud file
        :return: surface point cloud points
    """
    # From bulk cloud extracts only the points on the surface
    file_path = 'models/{}/foot_bulk.npy'.format(directory)
    Z = np.load(file_path)
    pts_per_unit = 5
    X0 = np.array(np.round((Z[:, 0]-np.min(Z[:, 0]))*pts_per_unit)).astype(int).reshape(-1, 1)
    X1 = np.array(np.round((Z[:, 1]-np.min(Z[:, 1]))*pts_per_unit)).astype(int).reshape(-1, 1)
    X2 = np.array(np.round((Z[:, 2]-np.min(Z[:, 2]))*pts_per_unit)).astype(int).reshape(-1, 1)
    X = np.concatenate((X0, X1, X2), axis=1)
    x0_min, x0_max, x1_min, x1_max, x2_min, x2_max = np.min(Z[:, 0]), np.max(Z[:, 0]), np.min(Z[:, 1]), \
                                                     np.max(Z[:, 1]), np.min(Z[:, 2]), np.max(Z[:, 2])
    # Number of points as indexes
    x0_range = int(np.round((x0_max-x0_min)*pts_per_unit+1))
    x1_range = int(np.round((x1_max-x1_min)*pts_per_unit+1))
    x2_range = int(np.round((x2_max-x2_min)*pts_per_unit+1))
    # Define cloud with indexes
    n = np.zeros((x0_range, x1_range, x2_range))
    init = time.time()
    for i in X:
        try:
            n[i[0], i[1], i[2]] = 1
        except:
            pass
    # Extract the point lengths
    x0_len = len(n[:, 0, 0])
    x1_len = len(n[0, :, 0])
    x2_len = len(n[0, 0, :])
    # 2x2x2
    h=[]
    for x0 in range(x0_len):
        for x1 in range(x1_len):
            for x2 in range(x2_len):
                try:
                    if (n[x0, x1, x2] == 1) | (n[x0, x1, x2] == 2):
                        if (x0 == x0_range - 1) | (x0 == 0):    # 46 is the lower bound (surface of foot)
                            if all(((t == 1) | (t == 2)) for t in n[x0, x1 - 1:x1 + 2, x2].ravel()) & \
                                    all(((t == 1) | (t == 2)) for t in n[x0, x1, x2 - 1:x2 + 2].ravel()):
                                n[x0, x1, x2] = 2
                                h.append([x0/pts_per_unit+x0_min, x1/pts_per_unit+x1_min, x2/pts_per_unit+x2_min])
                                continue
                        if (x1 == x1_range - 1) | (x1 == 0):    # 46 is the lower bound (surface of foot)
                            if all(((t == 1) | (t == 2)) for t in n[x0 - 1:x0 + 2, x1, x2].ravel()) & \
                                    all(((t == 1) | (t == 2)) for t in n[x0, x1, x2 - 1:x2 + 2].ravel()):
                                n[x0, x1, x2] = 2
                                h.append([x0/pts_per_unit+x0_min, x1/pts_per_unit+x1_min, x2/pts_per_unit+x2_min])
                                continue
                        if (x2 == x2_range - 1) | (x2 == 0):    # 46 is the lower bound (surface of foot)
                            if all(((t == 1) | (t == 2)) for t in n[x0 - 1:x0 + 2, x1, x2].ravel()) & \
                                    all(((t == 1) | (t == 2)) for t in n[x0, x1 - 1:x1 + 2, x2].ravel()):
                                n[x0, x1, x2] = 2
                                h.append([x0/pts_per_unit+x0_min,x1/pts_per_unit+x1_min,x2/pts_per_unit+x2_min])
                                continue
                        if any(t == 0 for t in n[x0 - 1:x0 + 2, x1, x2].ravel()) | \
                                any(t == 0 for t in n[x0, x1 - 1:x1 + 2, x2].ravel()) | \
                                any(t == 0 for t in n[x0, x1, x2 - 1:x2 + 2].ravel()):
                            n[x0, x1, x2] = 2
                            h.append([x0/pts_per_unit+x0_min, x1/pts_per_unit+x1_min, x2/pts_per_unit+x2_min])
                except:
                    continue
    end = time.time()
    h = np.float64(h)
    Z = h
    #print('Cloud size: {}'.format(len(Z)))
    #print('Timestamp: {}'.format(end-init))
    # Include paper corners
    obj = np.float64([[-29/2, -21/2, 0], [29/2, -21/2, 0], [-29/2, 21/2, 0], [29/2, 21/2, 0]])
    Z2 = np.ones(len(Z)).reshape(1, -1)
    Z = np.transpose(Z)
    Z3 = np.concatenate((Z, Z2), axis=0)
    np.save('models/{}/surface_cloud.npy'.format(directory), Z3)
    return Z3


def parameters(directory):
    """
        Get foot parameters from the surface point cloud
        :param directory: input path directory for surface point cloud
        :return: length and width of the foot surface
    """
    # Get surface file
    surface = 'models/{}/surface_cloud.npy'.format(directory)
    # Get length and width of foot
    Z = np.load(surface)
    Z = np.transpose(Z)
    X = Z[:,:2]
    rho = 1
    comp = 0
    larg = np.Infinity
    final_angle_comp = 0
    final_angle_larg = 0
    for angle in range(0, 180):
        angle = angle*np.pi/180
        x = rho*np.cos(angle)
        y = rho*np.sin(angle)
        proj = np.float64([x, y])
        proj = proj/np.linalg.norm(proj)
        F = np.dot(X, proj.T)
        a = np.max(F)-np.min(F)
        if a > comp:
            comp = a
            final_angle_comp = angle
        if a < larg:
            larg = a
            final_angle_larg = angle
        angle = 95.0
        angle = angle*np.pi/180
        x = rho*np.cos(angle)
        y = rho*np.sin(angle)
        proj = np.float64([x, y])
        proj = proj/np.linalg.norm(proj)
        F = np.dot(X, proj.T)
        a = np.max(F)-np.min(F)
    # Compute foot arc value based on the sheet radius
    radius = larg/2
    footArc = math.sqrt(4*math.sqrt(4*math.pow(radius, 2) - (math.pow(larg, 2)/2))) + radius
    #print('Height: {0:.3f}, angle: {1:.1f}'.format(comp, final_angle_comp*180/np.pi))
    #print('Width: {0:.3f}, angle: {1:.1f}'.format(larg, final_angle_larg*180/np.pi))
    #print('Foot Arc: {0:.3f}'.format(footArc))
    file_ = open("sheet_dimensions.txt", "w")
    file_.write('Height: {0:.1f} cm\n'.format(comp))
    file_.write('Width: {0:.1f} cm\n'.format(larg))
    file_.write('Foot Arc: {0:.1f} cm'.format(footArc))

