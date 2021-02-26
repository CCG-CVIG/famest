# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
import numpy as np
import cv2
import math
import time
import os
import gc


# ------------------------
#   FUNCTIONS
# ------------------------
def point_dist(p1, p2):
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return distance

def midpoint(p1, p2):
    return [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]

def get_array_points(pts, new_img):

    new_pts = pts
    if point_dist(pts[0], pts[1]) > point_dist(pts[0], pts[2]):
        print("hdsfsfhsfhsfhsjfhsfhs")
        new_pts = [pts[0], pts[2], pts[1], pts[3]]
    pts = new_pts

    m01 = midpoint(pts[0], pts[1])
    m23 = midpoint(pts[2], pts[3])

    # Draw circle in paper point coordinates
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    i = 0
    for pt in pts:
        cv2.circle(new_img, (pt[0], pt[1]), 36, color_list[i], -1)
        i += 1
    cv2.circle(new_img, (m01[0], m01[1]), 36, (255, 0, 255), -1)
    cv2.circle(new_img, (m23[0], m23[1]), 36, (255, 0, 255), -1)
    show_image(new_img, "ImagePoints")

    angle01 = math.atan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0]) * (180 / math.pi)
    angle23 = math.atan2(pts[2][1] - pts[3][1], pts[2][0] - pts[3][0]) * (180 / math.pi)
    angleM = math.atan2(m01[1] - m23[1], m01[0] - m23[0]) * (180 / math.pi)
    print(angle01, angle23, angleM)

    if angleM > 90 or angleM < -90:
        if angle01 >= 0:
            if angle23 >= 0:
                points = np.array([pts[1], pts[3], pts[0], pts[2]])
            else:
                points = np.array([pts[1], pts[2], pts[0], pts[3]])
        else:
            if angle23 >= 0:
                points = np.array([pts[0], pts[3], pts[1], pts[2]])
            else:
                points = np.array([pts[0], pts[2], pts[1], pts[3]])
    else:
        if angle01 >= 0:
            if angle23 >= 0:
                points = np.array([pts[0], pts[2], pts[1], pts[3]])
            else:
                points = np.array([pts[0], pts[3], pts[1], pts[2]])
        else:
            if angle23 >= 0:
                points = np.array([pts[1], pts[2], pts[0], pts[3]])
            else:
                points = np.array([pts[1], pts[3], pts[0], pts[2]])

    return points

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

def select_lines(lines, error=0.1):
    """
        Select lines according to a error percentage
        :param lines: list of lines
        :param error: error percentage
        :return:
    """
    def dist(a, b):
        """
            Get the distance between two points: a and b.
            :param a: point a
            :param b: point b
            :return: distance between point a and point b
        """
        a_ = a.copy()
        b_ = b.copy()
        if a_[0][1] > np.pi:
            a_[0][1] -= np.pi
        if a_[0][1] < 0:
            a_[0][1] += np.pi
        if b_[0][1] > np.pi:
            b_[0][1] -= np.pi
        if b_[0][1] < 0:
            b_[0][1] += np.pi
        p0 = (a_[0][0] - b_[0][0]) ** 2
        p1 = (a_[0][1] - b_[0][1]) ** 2
        p = p0 * p1 / 1000
        return p

    lst = []
    for line in lines:
        if line[0][0] < 0:
            line[0][0] = -line[0][0]
            line[0][1] = line[0][1] - np.pi
        check = True
        for l in lst:
            if dist(line, l) < error:
                check = False
        if check:
            lst.append(line)
    return np.array(lst)


def resize(img1, img2, fsize=0.2, a='small'):
    """
        Resize on image size according to another.
        :param img1: First image input
        :param img2: Second image input
        :param fsize: Final image size
        :param a: Image Size Flag
        :return: Resized Image
    """
    _img = img1.copy()
    if img2 is not None:
        _img2 = img2.copy()
    # Resize the image from small to big, and big to small
    if a == 'small':
        _img = cv2.resize(_img, (int(_img.shape[1] * fsize), int(_img.shape[0] * fsize)),
                          interpolation=cv2.INTER_LINEAR)
    if a == 'big':
        _img = cv2.resize(_img, (_img2.shape[1], _img2.shape[0]), interpolation=cv2.INTER_LINEAR)
    return _img


def paper(img):
    """
        Returns the paper corners from an image
        :param img: input image
        :return: paper corners
    """
    def get_Area(cnt):
        """
            Get contour area
            :param cnt: list of contours
            :return: contour area
        """
        return cv2.contourArea(cnt)

    def cnt_paper(img):
        """
            Get paper contours
            :param img: input image
            :return: list of paper contours
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)
        img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 351, 4)
        canny = cv2.Canny(img_gray, 50, 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
        canny = cv2.dilate(canny, kernel)
        kernel = np.ones((5, 5), np.uint8)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        canny = cv2.erode(canny, kernel2, iterations=7)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canny = cv2.erode(canny, kernel3, iterations=20)
        cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=get_Area, reverse=True)
        return cnts

    def select_cnt(img, cnts):
        """
            Select contours
            :param img: input image
            :param cnts: list of contours
            :return: contour
        """
        def k(item):
            """
                Get k mean from stats_cnt using an item on the image
                :param item: item
                :return: k mean from stats_cnt
            """
            return np.mean(stats_cnt(img, item)[0])
        temp_cnts = []
        cnts = [cnt for cnt in cnts if ((proportion_area(img, cnt) > 0.02) & (proportion_area(img, cnt) < 0.8))]
        for cnt in cnts:
            mean, std = stats_cnt(img, cnt)
            # #print("Mean: ", (mean, std))
            # #print("Diff: ", (np.max(mean) - np.min(mean)))
            # #print("Max. Std: ", np.max(std))
            # #print("Min. Med: ", np.min(mean))
            # #print("Percentile: ", np.percentile(np.mean(mean), 90))
            # #print("Mean Mean: ", np.mean(mean))
            if (np.max(mean) - np.min(mean) < 60) and \
                    (np.max(std) < 60) and \
                    (np.min(mean) > 0.7 * np.percentile(np.mean(mean), 90)) and \
                    (np.mean(mean) > 120):
                temp_cnts.append(cnt)
        cnts = temp_cnts
        # for i, cnt in zip(range(len(cnts)), cnts):
        #     aux_img = img.copy()
        #     cv2.drawContours(aux_img, [cnt], -1, (0, 0, 255), -1)
        #     show_image(aux_img, "Image {}".format(i))
        cnts = sorted(cnts, key=k, reverse=True)
        return cnts[0]

    def stats_cnt(img, cnt):
        """
            Returns the status of each one of the contours
            :param img: input image
            :param cnt: contour
            :return: status of each one of the contours
        """
        if len(img.shape) > 2:
            mask = np.zeros_like(img[:, :, 0])
        else:
            mask = np.zeros_like(img)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        return cv2.meanStdDev(img, mask=mask)

    def find_centroid(cnt):
        """
            Find the centroid
            :param cnt: contour
            :return: centroid ((x,y), x -> , y |)
        """
        M = cv2.moments(cnt)
        cX = int(round(M['m10'] / M['m00']))
        cY = int(round(M['m01'] / M['m00']))
        return cX, cY

    def dist_to_center(img, cnt):
        """
            Return the distance to center of the image
            :param img: input image
            :param cnt: contour
            :return: distance to the center
        """
        y_img, x_img = img.shape[0:2]
        center = (x_img, y_img)
        centroid = find_centroid(cnt)
        return np.sqrt((center[0] - centroid[0]) ** 2 + (center[1] - centroid[1]) ** 2)

    def proportion_area(img, cnt):
        """
            Get the proportion area in the image
            :param img: input image
            :param cnt: contour
            :return: proportion area
        """
        rows, cols = img.shape[0:2]
        total_area = rows * cols
        cnt_area = get_Area(cnt)
        return cnt_area / total_area

    def k(item):
        """
            Get contour area based on a contour from the image
            :param item: contour item
            :return: contour area
        """
        return cv2.contourArea(item)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #show_image(img_gray, "Original Gray")
    img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)
    #show_image(img_gray, "After Bilateral Filter")
    img_gray = clahe.apply(img_gray)
    #show_image(img_gray, "CLAHE Gray")
    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 351, 4)
    canny = cv2.Canny(img_gray, 50, 20)

    # Eliminate points
    img = cv2.bilateralFilter(img, 9, 75, 75)
    canny2 = cv2.Canny(img, 100, 120)
    kernel = np.ones((3, 3), np.uint8)
    canny2 = cv2.dilate(canny2, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    canny = cv2.dilate(canny, kernel)
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    canny = cv2.erode(canny, kernel2, iterations=7)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    canny = cv2.erode(canny, kernel3, iterations=20)

    # Find contours
    cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=k, reverse=True)
    mask = np.zeros_like(canny)
    cv2.drawContours(mask, [cnts[1]], -1, 255, -1)
    cnt1_convex = cv2.convexHull(cnts[1], True)
    cnt1 = cnts[1]
    cv2.drawContours(mask, [cnt1_convex], -1, 255, -1)
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    img_ = grabCut(img.copy(), mask.copy(), iterations=2)
    mask = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img_.copy(), cv2.COLOR_BGR2GRAY)
    img_ = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    mask2 = np.zeros_like(canny.copy())
    stats_cnt(img, cnt1)
    cnt_ = select_cnt(img, cnts)
    cv2.drawContours(mask2, [cnt_], -1, 255, -1)
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    return mask2

def grabCut(img, mask_foot, iterations):
    """
        Grab a cut portion of the image according to a given mask
        :param img: input image
        :param mask_foot: input foot mask
        :param iterations: number of iterations
        :return: grab cut image
    """
    # Apply grabCut to img based on the mask given
    _img = img.copy()
    _img_resized = resize(_img, None)
    mask_foot_resized = resize(mask_foot, None)
    mask = np.zeros(_img_resized.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    newmask = mask_foot_resized
    newmask = cv2.cvtColor(mask_foot_resized, cv2.COLOR_BGR2GRAY)

    mask[newmask < 20] = cv2.GC_PR_BGD
    mask[newmask > 240] = cv2.GC_PR_FGD
    mask[(newmask >= 20) & (newmask <= 240)] = cv2.GC_PR_BGD

    mask, bgdModel, fgdModel = cv2.grabCut(_img_resized, mask, None, bgdModel, fgdModel, iterations,
                                           cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    _img_resized = _img_resized * mask[:, :, np.newaxis]
    _img = resize(_img_resized, _img, a='big')

    # Denoysing with erosion followed by dilation
    kernel = np.ones((5, 5), np.uint8)
    _img = cv2.erode(_img, kernel, iterations=3)
    _img = cv2.dilate(_img, kernel, iterations=3)
    return _img


def cart2pol(x, y):
    """
        Convert the cartesian coordinates into polar coordinates
        :param x: cartesian coordinate x
        :param y: cartesian coordinate y
        :return: polar coordinates (rho,phi)
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """
        Convert the polar coordinates into cartesian coordinates
        :param rho: rho polar coordinate
        :param phi: phi polar coordinate
        :return: cartesian coordinates (x,y)
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def intersection_from_pairs(pair1, pair2):
    """
        Calculate the intersection between two pairs of points
        :param pair1: first pair of points
        :param pair2: second pair of points
        :return: two pair intersection
    """
    def line(p1, p2):
        """
            Get a line based on two points (p1, p2)
            :param p1: point p1
            :param p2: point p2
            :return: line drawn between point p1 and p2
        """
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    L1, L2 = line(pair1[0], pair1[1]), line(pair2[0], pair2[1])
    x0, y0, b0 = L1
    x1, y1, b1 = L2
    # Check if lines are vertical or parallel
    if (y0 == 0) & (y1 == 0):
        return False
    if y0 == 0:
        x, y = -b0 / x0, (x1 * b0 - b1 * x0) / (y1 * x0)
    if y1 == 0:
        x, y = -b1 / x1, (x0 * b1 - b0 * x1) / (y0 * x1)
    if x0 / y0 == x1 / y1:
        return False
    elif x0 / y0 >= x1 / y1:
        x = -(b1 / y1 - b0 / y0) / (x0 / y0 - x1 / y1)
        y = -x1 / y1 * x + b1 / y1
        return np.array([int(round(x)), int(round(y))]).astype('int32')
    elif x0 / y0 <= x1 / y1:
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return np.array([int(round(x)), int(round(y))]).astype('int32')
    else:
        return False


def sort_pts(pts, centroid_all, centroid_convex):
    """
        Sort points using a centroid and a convex
        :param pts: list of points
        :param centroid_all: centroid
        :param centroid_convex:
        :return: list of sorted points
    """
    def dist(pt1, pt2):
        """
            Get distance between two points (pt1, pt2)
            :param pt1: point pt1
            :param pt2: point pt2
            :return: distance between the two points
        """
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def k(item):
        """
            Get distance between centroid and the centroid convex
            :param item: contour
            :return: distance between centroids
        """
        return dist(item, centroid_all) - dist(item, centroid_convex)
    pts_sorted = sorted(pts, key=k, reverse=False)
    return pts_sorted

def extract_metrics(img_, id, last_angle01, last_angle23):
    """
        Extract the metrics from the image
        :param img_: input image
        :return: extracted metrics
    """

    def k(item):
        """
            Get contour area from a contour item
            :param item: contour
            :return: contour area of a contour
        """
        return cv2.contourArea(item)
    img = img_.copy()
    img_copy = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create mask to segment paper
    mask = paper(img)
    # Use mask to segment, using grabCut, the paper
    im = grabCut(img, mask, 5)
    # Find contour of paper
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #print("[INFO] Detecting the corners")
    # -------------------------------------------------
    #   Corner Sub Pixel With Foot and Paper Separation
    # -------------------------------------------------
    ret, thresh = cv2.threshold(im_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = k, reverse=True)
    cnt = contours[0]

    def centroid(cnt):
        """
            Get centroid based on contour
            :param cnt: contour
            :return: centroid coordinates
        """
        M_ = cv2.moments(cnt)
        cx_ = int(M_['m10'] / M_['m00'])
        cy_ = int(M_['m01'] / M_['m00'])
        centroid_ = (cx_, cy_)
        return centroid_

    # Get convexhull of paper
    conv_hull = cv2.convexHull(cnt, returnPoints=True)
    # #print("Convex Hull 01: ", conv_hull)
    # Get centroid from contour and from convexhull of paper
    centroid_cnt = centroid(cnt)
    centroid_convex = centroid(conv_hull)
    c = np.array(contours[0])[:, 0, :]
    c1 = c[:, 0]
    c2 = c[:, 1]
    cnt = contours[0]
    # Get convexHull with 4 or 5 points (corners)
    # (If 5 points are found, we have to find the hidden point)
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    conv_hull2 = cv2.convexHull(approx, returnPoints=True)
    # #print("Convex Hull 02: ", conv_hull2)
    epsilon = 0.001 * cv2.arcLength(conv_hull2, True)
    conv_hull3 = cv2.approxPolyDP(conv_hull2, epsilon, True)
    # #print("Convex Hull 03: ", conv_hull3)
    n = 0.001
    while len(conv_hull3) > 5:
        epsilon = (0.001 + n) * cv2.arcLength(conv_hull2, True)
        conv_hull3 = cv2.approxPolyDP(conv_hull2, epsilon, True)
        n += 0.001
    # Remove points very close to corners
    conv_hull4 = conv_hull3.copy()
    while len(conv_hull4) > 4:
        epsilon = (0.001 + n) * cv2.arcLength(conv_hull2, True)
        conv_hull4 = cv2.approxPolyDP(conv_hull2, epsilon, True)
        n += 0.001
    # #print("Convex Hull 04: ", conv_hull4)
    if np.abs(cv2.contourArea(conv_hull3) - cv2.contourArea(conv_hull4)) < 0.001 * cv2.contourArea(conv_hull3):
        conv_hull3 = conv_hull4
    # #print("Convex Hull 05: ", conv_hull3)
    # Find Orientation of points
    pts = sort_pts(conv_hull3[:, 0, :], centroid_cnt, centroid_convex)
    #print("Number of pts: {}".format(len(pts)))
    ############################################################################
    # new_img = img_.copy()
    # for pt in pts:
    #     #print(pt)
    #     cv2.circle(new_img, (pt[0], pt[1]), 63, (0, 0, 255), -1)
    # show_image(new_img, "ImagePoints")
    ############################################################################
    # Check if points are close to each other
    break_now = False
    for i in range(0, len(pts)):
        pt1 = pts[i]
        for j in range(i + 1, len(pts)):
            pt2 = pts[j]
            distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
            if distance < 100:
                #print("Distance between {} and {}".format(pt1, pt2) + ": {}".format(distance))
                #print("Points {} and {} are close to each other".format(pt1, pt2))
                del pts[j]
                break_now = True # check break flag
                break
        if break_now: # if break flag is true break from for loop
            break

    print(len(pts))
    if len(pts) == 5: # If 5 points we found
        conv_hull_list = list(conv_hull3[:, 0, :])
        special_argpt = [i for i, el in enumerate(conv_hull_list)
                         if ((pts[0][0] == conv_hull_list[i][0]) and (pts[0][1] == conv_hull_list[i][1]))]
        wrong_points1 = [conv_hull3[:, 0, :][special_argpt[0] - 2], conv_hull3[:, 0, :][special_argpt[0] - 1]]
        wrong_points2 = [conv_hull3[:, 0, :][special_argpt[0] - 4], conv_hull3[:, 0, :][special_argpt[0] - 3]]

        pt = intersection_from_pairs(wrong_points1, wrong_points2)
        print(pt)
        if centroid_convex[0] < pts[0][0]:
            points = np.array([conv_hull3[:, 0, :][special_argpt[0]], conv_hull3[:, 0, :][special_argpt[0] - 4],
                               conv_hull3[:, 0, :][special_argpt[0] - 1], pt])
            print("centroid lower")
        else:
            points = np.array([conv_hull3[:, 0, :][special_argpt[0] - 4], pt, conv_hull3[:, 0, :][special_argpt[0]],
                               conv_hull3[:, 0, :][special_argpt[0] - 1]])

        # Get array of points
        points = get_array_points(points, img_.copy())

    elif len(pts) == 4: # If only 4 points were found
        # Get array of points
        points = get_array_points(pts, img_.copy())

        # if (pts[0][0] < pts[1][0]):
        #     print("pts[0][0] < pts[1][0]")
        #     points = np.array([pts[0], pts[2], pts[1], pts[3]])
        # else:
        #     print("pts[0][0] >= pts[1][0]")
        #     points = np.array([pts[1], pts[3], pts[0], pts[2]])

    paper_points = points.copy()
    # point_order = [0, 1, 2, 3]
    # for i, pt in zip(range(0, len(paper_points)), paper_points):
    #     paper_points[i] = points[point_order[i]]

    # Segment Foot
    mask_foot = np.zeros_like(gray)
    mask_foot = cv2.drawContours(mask_foot, [conv_hull3], -1, 255, -1)
    mask_foot = cv2.drawContours(mask_foot, [cnt], -1, 0, -1)
    mask_contours, _ = cv2.findContours(mask_foot, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask_cnt = sorted(mask_contours, key=k, reverse=True)[0]
    mask_foot = np.zeros_like(gray)
    mask_foot = cv2.drawContours(mask_foot, [mask_cnt], -1, 255, -1)
    mask_foot2 = np.zeros_like(mask_foot)
    mask_foot2[(gray < 50) & (mask_foot == 255)] = 255
    mask_foot2 = cv2.cvtColor(mask_foot2, cv2.COLOR_GRAY2BGR)
    im2 = grabCut(img_copy, mask_foot2, 3)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    foot_segmented = np.zeros_like(gray)
    foot_segmented[im2_gray > 0] = 255

    def fill_holes(img_th):
        """
            Fill holes in order to remove them from the foot mask
            :param img_th: foot mask
            :return: flood filled image
        """
        # Copy the thresholded image.
        img_floodfill = img_th.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = img_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(img_floodfill, mask, (0, 0), 255)
        # Invert floodfilled image
        img_floodfill_inv = cv2.bitwise_not(img_floodfill)
        # Combine the two images to get the foreground.
        img_out = img_th | img_floodfill_inv
        #show_image(img_out, "Mask")
        return img_out

    return paper_points, fill_holes(foot_segmented), last_angle01, last_angle23
