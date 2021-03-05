# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import cv2
import math
import time
import os
import gc
# from source.foot_data import plot_image

# ------------------------
#   GLOBAL VARIABLES
# ------------------------
last_angleM = -1000000
last_angle02 = -1000000
to_change = False


# ----------------------------------------
#   FUNCTIONS
# ----------------------------------------
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


def point_distance(pt1, pt2):
    """
        Get the distance between two points
        :param pt1: first point
        :param pt2: second point
        :return: distance between the two points
    """
    distance = math.sqrt(abs(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2)))
    return distance


def midpoint(pt1, pt2):
    """
        Find the midpoint between two points
        :param pt1: first point
        :param pt2: second point
        :return: midpoint between two points
    """
    return [int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)]


def verify_coords(points):
    """
        Verify if the set of coordinates is correct
        :param points: set of points
        :return: boolean
    """
    valid_coords = True
    for pt in points:
        if pt[0] < 0 or pt[1] < 0:
            valid_coords = False
            break
    return valid_coords


def draw_paper_circles(pts, img):
    """
        Draw circle in paper point coordinates
        :param pts: set of paper point coordinates
        :param img: input image
        :return:
    """
    color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    i = 0
    for pt in pts:
        cv2.circle(img, (pt[0], pt[1]), 36, color_lst[i], -1)
        i += 1
    #show_image(img, "ImagePoints")


def draw_paper_circles_with_midpoints(pts, img, midpts):
    """
        Draw circles to paper point coordinates with midpoints
        :param pts: set of paper point coordinates
        :param img: input image
        :param midpts: set of midpoints
        :return:
    """
    color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    i = 0
    # Draw circles in paper points
    for pt in pts:
        cv2.circle(img, (pt[0], pt[1]), 36, color_lst[i], -1)
        i += 1
    # Draw circles in midpoints
    for midpt in midpts:
        cv2.circle(img, (midpt[0], midpt[1]), 36, (255, 0, 255), -1)
    #show_image(img, "ImagePoints")


def draw_paper_circles_with_multipoints(pts, img, midpts1, midpts2, centerpts, xyzpts):
    """
        Draw circles to paper point coordinates with midpoints
        :param pts: set of paper point coordinates
        :param img: input image
        :param midpts1: first set of midpoints
        :param midpts2: second set of midpoints
        :param centerpts: set of center points
        :param xyzpts: set of extra points
        :return:
    """
    color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    pts_txt = ["B0", "G1", "R2", "Y3"]
    midpts1_txt = ["M1", "M3"]
    midpts2_txt = ["M2", "M4"]
    xyzw_txt = ["X", "Y", "Z", "W"]
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1
    TEXT_THICKNESS = 2
    i = 0
    j = 0
    k = 0
    m = 0
    # Draw circles in paper points
    for pt in pts:
        print(i, color_lst[i])
        cv2.circle(img, (pt[0], pt[1]), 36, color_lst[i], -1)
        text_size, _ = cv2.getTextSize(pts_txt[i], TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(pt[0] - text_size[0] / 2), int(pt[1] + text_size[1] / 2))
        cv2.putText(img, pts_txt[i], text_origin, TEXT_FACE, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
        i += 1
    # Draw circles in the first set midpoints
    for pt in midpts1:
        cv2.circle(img, (pt[0], pt[1]), 36, (255, 0, 255), -1)
        text_size, _ = cv2.getTextSize(midpts1_txt[j], TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(pt[0] - text_size[0] / 2), int(pt[1] + text_size[1] / 2))
        cv2.putText(img, midpts1_txt[j], text_origin, TEXT_FACE, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
        j += 1
    # Draw circles in the second set midpoints
    for pt in midpts2:
        cv2.circle(img, (pt[0], pt[1]), 36, (255, 255, 0), -1)
        text_size, _ = cv2.getTextSize(midpts2_txt[k], TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(pt[0] - text_size[0] / 2), int(pt[1] + text_size[1] / 2))
        cv2.putText(img, midpts2_txt[k], text_origin, TEXT_FACE, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
        k += 1
    # Draw circles in the set of center points
    for pt in centerpts:
        cv2.circle(img, (pt[0], pt[1]), 36, (125, 0, 125), -1)
        text_size, _ = cv2.getTextSize("C", TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(pt[0] - text_size[0] / 2), int(pt[1] + text_size[1] / 2))
        cv2.putText(img, "C", text_origin, TEXT_FACE, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
    # Draw circles in the set of center points
    for pt in xyzpts:
        cv2.circle(img, (pt[0], pt[1]), 36, (125, 125, 0), -1)
        text_size, _ = cv2.getTextSize(xyzw_txt[m], TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(pt[0] - text_size[0] / 2), int(pt[1] + text_size[1] / 2))
        cv2.putText(img, xyzw_txt[m], text_origin, TEXT_FACE, TEXT_SCALE, (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
        m += 1
    #show_image(img, "ImagePoints")


def draw_contour_bbox(contours, img):
    """
        Given a list of contours draw a bounding box and make sure the contours are in the rectangle
        :param contours: list of contours
        :param img: input image
        :return:
    """
    color_lst = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    i = 0
    j = 0
    print("Number of contours = {}".format(str(len(contours))))
    for cnt in contours:
        print(i, color_lst[i])
        if cv2.contourArea(cnt) > 15:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), color_lst[i], 3)
            i += 1
    # Draw each one of the contour with a different colour
    for cnt in contours:
        print(j, color_lst[j])
        cv2.drawContours(img, cnt, -1, color_lst[j], 3)
        j += 1
    #show_image(img, "Contours")


def get_front_midpoint_angles(pts):
    """
        Get the midpoint angles from a set of points
        :param pts: get the set of points
        :return: midpoint angles
    """
    m01 = midpoint(pts[0], pts[1])
    m23 = midpoint(pts[2], pts[3])
    angle01 = 180 - math.atan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0]) * (180 / math.pi)
    angle23 = 180 - math.atan2(pts[2][1] - pts[3][1], pts[2][0] - pts[3][0]) * (180 / math.pi)
    angleM = 180 - math.atan2(m01[1] - m23[1], m01[0] - m23[0]) * (180 / math.pi)
    #print("Midpoints (1): ", [m01, m23])
    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23) + ", AngleM: {}".format(angleM))
    return m01, m23, angle01, angle23, angleM


def get_side_midpoint_angles(pts):
    """
        Get the midpoint angles from a set of points
        :param pts: get the set of points
        :return: midpoint angles
    """
    m02 = midpoint(pts[0], pts[2])
    m13 = midpoint(pts[1], pts[3])
    angle02 = 180 - math.atan2(pts[0][1] - pts[2][1], pts[0][0] - pts[2][0]) * (180 / math.pi)
    angle13 = 180 - math.atan2(pts[1][1] - pts[3][1], pts[1][0] - pts[3][0]) * (180 / math.pi)
    angleM = 180 - math.atan2(m02[1] - m13[1], m02[0] - m13[0]) * (180 / math.pi)
    #print("Midpoints (2): ", [m02, m13])
    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13) + ", AngleM: {}".format(angleM))
    return m02, m13, angle02, angle13, angleM


def get_angleM_diff(angleM, last_angleM):
    """
        Get the difference between angleMs
        :param angleM: angle between the two midpoints
        :param last_angleM: last angle between midpoints
        :return: angle difference
    """
    if last_angleM == -1000000:
        angleM_diff = 0
    else:
        angleM_diff = abs(last_angleM - angleM)
        if angleM_diff > 180:
            angleM_diff = abs(angleM_diff - 360)
    #print("Diff: ", angleM_diff)
    #print("Last Angle: ", last_angleM)
    return angleM_diff


def get_new_coords(pts, index):
    """
        Get new coordinate order
        :param pts: set of points
        :param index: index order
        :return: new coordinate order
    """
    if index == 3:
        mp = midpoint(pts[1], pts[2])
        new_x = 2 * mp[0] - pts[0][0]
        new_y = 2 * mp[1] - pts[0][1]
    else:
        mp = midpoint(pts[0], pts[3])
        new_x = 2 * mp[0] - pts[2][0]
        new_y = 2 * mp[1] - pts[2][1]
    return [new_x, new_y]


def get_array_points(pts, new_img, input_type):
    """
        Get array of points from an image
        :param pts: set of points
        :param new_img: input image
        :param input_type: type of input file
        :return: array of points
    """
    if input_type == "photo":
        new_pts = pts
        if point_distance(pts[0], pts[1]) > point_distance(pts[0], pts[2]):
            #print("hdsfsfhsfhsfhsjfhsfhs")
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
        #show_image(new_img, "ImagePoints")
        angle01 = math.atan2(pts[0][1] - pts[1][1], pts[0][0] - pts[1][0]) * (180 / math.pi)
        angle23 = math.atan2(pts[2][1] - pts[3][1], pts[2][0] - pts[3][0]) * (180 / math.pi)
        angleM = math.atan2(m01[1] - m23[1], m01[0] - m23[0]) * (180 / math.pi)
        #print(angle01, angle23, angleM)
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
        #print("#######################################################################################################")
        return points

    else:
        DIFF_THRESH = 60
        DIFF_THRESH2 = 90
        # Draw circle in paper coordinates
        # draw_paper_circles(pts, new_img)
        # Check points
        last_pts = pts.copy()
        new_pts = pts.copy()
        # Check point distance
        #print("Distance between 0-1: {}".format(point_distance(pts[0], pts[1])) + " and Distance between 0-2: {}".format(point_distance(pts[0], pts[2])))
        #print("Distance between 2-3: {}".format(point_distance(pts[2], pts[3])) + " and Distance between 2-0: {}".format(point_distance(pts[2], pts[0])))
        if point_distance(pts[0], pts[1]) > point_distance(pts[0], pts[2]) and \
                point_distance(pts[2], pts[3]) > point_distance(pts[2], pts[0]):
            #print("Passed here")
            new_pts = [pts[0], pts[2], pts[1], pts[3]]
        pts = new_pts.copy()
        # Get midpoints
        m1 = [(0, 0)]
        m2 = [(0, 0)]
        m3 = [(0, 0)]
        m4 = [(0, 0)]
        m13 = [(0, 0)]
        m24 = [(0, 0)]
        mx = [(0, 0)]
        my = [(0, 0)]
        mz = [(0, 0)]
        mw = [(0, 0)]
        # Get angles between points 0-1 and 2-3
        m01, m23, angle01, angle23, angleM = get_front_midpoint_angles(pts)
        m02, m13, angle02, angle13, angleN = get_side_midpoint_angles(pts)
        # Check global angles and check if they are within the threshold
        global last_angleM
        global last_angle02
        global to_change
        angleM_diff = get_angleM_diff(angleM, last_angleM)
        # angle02_diff = get_angleM_diff(angle02, last_angle02)
        recovered = False
        last_angleM_diff = angleM_diff
        # last_angle02_diff = angle02_diff
        #print("Angle M Diff: ", angleM_diff)
        if angleM_diff > DIFF_THRESH:
            #print("Recovered")
            recovered = True
            pts = last_pts
            m01, m23, angle01, angle23, angleM = get_front_midpoint_angles(pts)
            angleM_diff = get_angleM_diff(angleM, last_angleM)
        if ((angleM_diff > DIFF_THRESH and not to_change) or (angleM_diff <= DIFF_THRESH and to_change)) \
                and not recovered:
            #print("To Change")
            new_pts = [pts[2], pts[3], pts[0], pts[1]]
            to_change = True
            pts = new_pts
        elif angleM_diff > DIFF_THRESH and to_change:
            #print("Revert")
            to_change = False
        last_angleM = angleM
        # midpts = [m01, m23]
        # draw_paper_circles_with_midpoints(pts, new_img, midpts)
        if angleM < 90 or angleM > 270:
            if angle01 <= 180:
                if angle23 <= 180:
                    points = np.array([pts[1], pts[3], pts[0], pts[2]])
                    #print("Point Order -> 1-3-0-2 (1)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[0], pts[1])
                    m2 = midpoint(pts[1], pts[3])
                    m3 = midpoint(pts[3], pts[2])
                    m4 = midpoint(pts[2], pts[0])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
                else:
                    points = np.array([pts[1], pts[2], pts[0], pts[3]])
                    #print("Point Order -> 1-2-0-3 (1)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[0], pts[1])
                    m2 = midpoint(pts[1], pts[2])
                    m3 = midpoint(pts[2], pts[3])
                    m4 = midpoint(pts[3], pts[0])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
            else:
                if angle23 <= 180:
                    points = np.array([pts[0], pts[3], pts[1], pts[2]])
                    #print("Point Order -> 0-3-1-2 (1)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[1], pts[0])
                    m2 = midpoint(pts[0], pts[3])
                    m3 = midpoint(pts[3], pts[2])
                    m4 = midpoint(pts[2], pts[1])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
                else:
                    # if angle02 > 120 and angle13 < 120:
                    #     points = np.array([pts[1], pts[2], pts[0], pts[3]])
                    #     print("Point Order -> 1-2-0-3 (1)")
                    #     print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #     print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    #     m1 = midpoint(pts[0], pts[1])
                    #     m2 = midpoint(pts[1], pts[2])
                    #     m3 = midpoint(pts[2], pts[3])
                    #     m4 = midpoint(pts[3], pts[0])
                    #     m13 = midpoint(m1, m3)
                    #     m24 = midpoint(m2, m4)
                    #     mx = midpoint(pts[0], m13)
                    #     my = midpoint(pts[1], m13)
                    #     mw = midpoint(pts[2], m13)
                    #     mz = midpoint(pts[3], m13)
                    # else:
                    points = np.array([pts[0], pts[2], pts[1], pts[3]])
                    #print("Point Order -> 0-2-1-3 (1)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[1], pts[0])
                    m2 = midpoint(pts[0], pts[2])
                    m3 = midpoint(pts[2], pts[3])
                    m4 = midpoint(pts[3], pts[1])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
        else:
            if angle01 <= 180:
                if angle23 <= 180:
                    points = np.array([pts[0], pts[2], pts[1], pts[3]])
                    #print("Point Order -> 0-2-1-3 (2)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[1], pts[0])
                    m2 = midpoint(pts[0], pts[2])
                    m3 = midpoint(pts[2], pts[3])
                    m4 = midpoint(pts[3], pts[1])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
                else:
                    points = np.array([pts[0], pts[3], pts[1], pts[2]])
                    #print("Point Order -> 0-3-1-2 (2)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[1], pts[0])
                    m2 = midpoint(pts[0], pts[3])
                    m3 = midpoint(pts[3], pts[2])
                    m4 = midpoint(pts[2], pts[1])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
            else:
                if angle23 < 180:
                    points = np.array([pts[1], pts[2], pts[0], pts[3]])
                    #print("Point Order -> 1-2-0-3 (2)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[0], pts[1])
                    m2 = midpoint(pts[1], pts[2])
                    m3 = midpoint(pts[2], pts[3])
                    m4 = midpoint(pts[3], pts[0])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
                else:
                    points = np.array([pts[1], pts[3], pts[0], pts[2]])
                    #print("Point Order -> 1-3-0-2 (2)")
                    #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23))
                    #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
                    m1 = midpoint(pts[0], pts[1])
                    m2 = midpoint(pts[1], pts[3])
                    m3 = midpoint(pts[3], pts[2])
                    m4 = midpoint(pts[2], pts[0])
                    m13 = midpoint(m1, m3)
                    m24 = midpoint(m2, m4)
                    mx = midpoint(pts[0], m13)
                    my = midpoint(pts[1], m13)
                    mw = midpoint(pts[2], m13)
                    mz = midpoint(pts[3], m13)
        if (angleM_diff > 120 and last_angleM_diff != angleM_diff) or (angleM > 180 and recovered):
            points = np.array([points[3], points[2], points[1], points[0]])
            #print("Point Order -> 3-2-1-0 (1)")
            #print("Angle01: {}".format(angle01) + ", Angle23: {}".format(angle23) + ", Angle Diff: {}".format(angleM_diff))
            #print("Angle02: {}".format(angle02) + ", Angle13: {}".format(angle13))
            m1 = midpoint(points[1], points[3])
            m2 = midpoint(points[3], points[2])
            m3 = midpoint(points[2], points[0])
            m4 = midpoint(points[0], points[1])
            m13 = midpoint(m1, m3)
            m24 = midpoint(m2, m4)
            mx = midpoint(pts[0], m13)
            my = midpoint(pts[1], m13)
            mw = midpoint(pts[2], m13)
            mz = midpoint(pts[3], m13)
            # If one of the points is at the wrong position, put it on the right spot
            # dist_01 = point_dist(points[0], points[1])
            # print("DIST 01: ", dist_01)
            # dist_23 = point_dist(points[2], points[3])
            # print("DIST 23: ", dist_23)
            # diff_dists = abs(dist_01 - dist_23)
            # print("DIFF DISTS: ", diff_dists)
            # if (diff_dists > dist_01 * 0.1) and (dist_01 < dist_23):
            #     points[1] = get_new_coords(points, 1)
            # elif diff_dists > dist_23 * 0.1 and (dist_23 < dist_01):
            #     points[3] = get_new_coords(points, 3)
        # Get midpoints
        midpts1 = [m1, m3]
        midpts2 = [m2, m4]
        centerpts = [m13, m24]
        xyzpts = [mx, my, mw, mz]
        #draw_paper_circles_with_multipoints(points, new_img, midpts1, midpts2, centerpts, xyzpts)
        #print("#######################################################################################################")
        return points


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
            if (np.max(mean) - np.min(mean) < 60) and \
                    (np.max(std) < 60) and \
                    (np.min(mean) > 0.7 * np.percentile(np.mean(mean), 90)) and \
                    (np.mean(mean) > 120):
                temp_cnts.append(cnt)
        cnts = temp_cnts
        cnts = sorted(cnts, key=k, reverse=True)
        if len(cnts) == 0:
            return []
        else:
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
    img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)
    img_gray = clahe.apply(img_gray)
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
    ok_flag = True
    if len(cnt_) == 0:
        ok_flag = False
    else:
        cv2.drawContours(mask2, [cnt_], -1, 255, -1)
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    return mask2, ok_flag


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
        :param centroid_convex: convex
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


def k(item):
    """
        Get contour area from contour item
        :param item: contour
        :return: contour area of a contour
    """
    return cv2.contourArea(item)


def extract_metrics(img_, input_type):
    """
        Extract the metrics from the image
        :param img_: input image
        :param model: input data model weights
        :param input_type: type of input file
        :return: extracted metrics
    """
    img = img_.copy()
    img_copy = img.copy()
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create mask to segment paper
    mask, ok_flag = paper(img)
    if not ok_flag:
        return None, None, False
    # Use mask to segment, using grabCut, the paper
    im = grabCut(img, mask, 5)
    # Find contour of paper
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print("[INFO] Detecting the corners")
    # -------------------------------------------------
    #   Corner Sub Pixel With Foot and Paper Separation
    # -------------------------------------------------
    ret, thresh = cv2.threshold(im_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_final = img_.copy()
    # Filter the contours
    for i in contours:
        size = cv2.contourArea(i)
        rect = cv2.minAreaRect(i)
        if size < 150000000:
            gray_np = np.float32(im_gray)
            mask = np.zeros(gray.shape, dtype="uint8")
            # To extract the corners separately for every shape you could first search for contours then apply
            # the Harris corner detection for each contour by drawing it on a mask with the cv2.fillPolly() function
            cv2.fillPoly(mask, [i], (255, 255, 255))
            dst = cv2.cornerHarris(mask, 20, 5, 0.04)
            ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            # corners = cv2.cornerSubPix(gray_np, np.float32(centroids), (5, 5), (-1, -1), criteria)
            corners = cv2.goodFeaturesToTrack(gray_np, 200, 0.01, 100)
            corners = np.int0(corners)
            # print("Corners Length: ", len(corners))
            if rect[2] <= 0:
                x, y, w, h = cv2.boundingRect(i)
                for idx in range(1, len(corners)):
                    # cX = np.max(corners[idx].item(0))
                    # cY = np.min(corners[idx].item(1))
                    cX, cY = corners[idx].ravel()
                    if x <= cX <= x + w and y <= cY <= y + h:
                        if cX <= cY:
                            # print("Foot ----> Green!")
                            # Draw green circles
                            # cv2.circle(img_, (int(corners[idx, 0]), int(corners[idx, 1])), 10, (0, 255, 0), 2)
                            cv2.circle(img_, (int(cX), int(cY)), 10, (0, 255, 0), 2)
                            # Threshold the image according to an optimal value
                            # img_[dst > 0.01 * dst.max()] = [0, 255, 0]  # Green
                        else:
                            # print("Paper ----> Blue!")
                            # Draw blue circles
                            # cv2.circle(img_, (int(corners[idx, 0]), int(corners[idx, 1])), 10, (255, 0, 0), 2)
                            cv2.circle(img_, (int(cX), int(cY)), 10, (255, 0, 0), 2)
                            # Threshold the image according to an optimal value
                            # img_[dst > 0.01 * dst.max()] = [255, 0, 0]  # BLUE
                    else:
                        # print("Out of Bounds! ----> Red!")
                        # # Draw red circles
                        # cv2.circle(img_, (int(corners[idx, 0]), int(corners[idx, 1])), 10, (0, 0, 255), 2)
                        cv2.circle(img_, (int(cX), int(cY)), 10, (0, 0, 255), 2)
                        # Threshold the image according to an optimal value
            img_[dst > 0.1 * dst.max()] = [0, 0, 255]  # RED
    contours = sorted(contours, key=k, reverse=True)
    if len(contours) != 0:
        cnt = contours[0]
    else:
        return None, None, False
    #draw_contour_bbox(contours, img_)
    # Get convexhull of paper
    conv_hull = cv2.convexHull(cnt, returnPoints=True)
    # Get centroid from contour and from convexhull of paper
    centroid_cnt = centroid(cnt)
    centroid_convex = centroid(conv_hull)
    c = np.array(contours[0])[:, 0, :]
    c1 = c[:, 0]
    c2 = c[:, 1]
    cnt = contours[0]
    # Get convexHull with 4 or 5 points (corners). (If 5 points are found, we have to find the hidden point)
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    conv_hull2 = cv2.convexHull(approx, returnPoints=True)
    epsilon = 0.001 * cv2.arcLength(conv_hull2, True)
    conv_hull3 = cv2.approxPolyDP(conv_hull2, epsilon, True)
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
    if np.abs(cv2.contourArea(conv_hull3) - cv2.contourArea(conv_hull4)) < 0.001 * cv2.contourArea(conv_hull3):
        conv_hull3 = conv_hull4
    # Find Orientation of points
    pts = sort_pts(conv_hull3[:, 0, :], centroid_cnt, centroid_convex)
    # Check if points are close to each other
    break_now = False
    for i in range(0, len(pts)):
        pt1 = pts[i]
        for j in range(i + 1, len(pts)):
            pt2 = pts[j]
            distance = math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
            if distance < 100:
                del pts[j]
                break_now = True
                break
        if break_now:
            break
    #print("Number of pts: {}".format(len(pts)))
    okay_flag = True
    # If 5 points we found
    if len(pts) == 5:
        conv_hull_list = list(conv_hull3[:, 0, :])
        special_argpt = [i for i, el in enumerate(conv_hull_list)
                         if ((pts[0][0] == conv_hull_list[i][0]) and (pts[0][1] == conv_hull_list[i][1]))]
        wrong_points1 = [conv_hull3[:, 0, :][special_argpt[0] - 2], conv_hull3[:, 0, :][special_argpt[0] - 1]]
        wrong_points2 = [conv_hull3[:, 0, :][special_argpt[0] - 4], conv_hull3[:, 0, :][special_argpt[0] - 3]]
        pt = intersection_from_pairs(wrong_points1, wrong_points2)
        if centroid_convex[0] < pts[0][0]:
            points = np.array([conv_hull3[:, 0, :][special_argpt[0]], conv_hull3[:, 0, :][special_argpt[0] - 4],
                               conv_hull3[:, 0, :][special_argpt[0] - 1], pt])
        else:
            points = np.array([conv_hull3[:, 0, :][special_argpt[0] - 4], pt, conv_hull3[:, 0, :][special_argpt[0]],
                               conv_hull3[:, 0, :][special_argpt[0] - 1]])
        # Get array of points
        if verify_coords(points):
            points = get_array_points(points, img_.copy(), input_type)
        else:
            okay_flag = False
    # If only 4 points were found
    elif len(pts) == 4:
        # Get array of points
        if verify_coords(pts):
            points = get_array_points(pts, img_.copy(), input_type)
        else:
            okay_flag = False
    paper_points = points.copy()
    # Segment Foot
    # mask_foot = np.zeros_like(gray)
    # mask_foot = cv2.drawContours(mask_foot, [conv_hull3], -1, 255, -1)
    # mask_foot = cv2.drawContours(mask_foot, [cnt], -1, 0, -1)
    # mask_contours, _ = cv2.findContours(mask_foot, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # mask_cnt = sorted(mask_contours, key=k, reverse=True)[0]
    # mask_foot = np.zeros_like(gray)
    # mask_foot = cv2.drawContours(mask_foot, [mask_cnt], -1, 255, -1)
    # mask_foot2 = np.zeros_like(mask_foot)
    # mask_foot2[(gray < 50) & (mask_foot == 255)] = 255
    # mask_foot2 = cv2.cvtColor(mask_foot2, cv2.COLOR_GRAY2BGR)
    # im2 = grabCut(img_copy, mask_foot2, 3)
    # im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # foot_segmented = np.zeros_like(gray)
    # foot_segmented[im2_gray > 0] = 255
    # foot_segmented, okay_flag = plot_image(img_final, model)
    # if foot_segmented is not None:
    #     #print(foot_segmented)
    #     foot_segmented = fill_holes(np.uint8(foot_segmented))
    #     #show_image(foot_segmented, "FOOT")
    # else:
    #     okay_flag = False
    # return paper_points, foot_segmented, okay_flag
    return paper_points, img_final
