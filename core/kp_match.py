import cv2
import time
import pydegensac
import numpy as np


def rord_matching(feat1, feat2):
    """
    rord matching
    :param image1:
    :param image2:
    :param feat1:
    :param feat2:
    :param debug:
    :return:
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(feat1['descriptors'], feat2['descriptors'])
    matches = sorted(matches, key=lambda x: x.distance)

    match1 = [m.queryIdx for m in matches]
    match2 = [m.trainIdx for m in matches]

    key_points_left = feat1['key_points'][match1, : 2]
    key_points_right = feat2['key_points'][match2, : 2]

    homo, inliers = pydegensac.findHomography(key_points_left, key_points_right, 10.0, 0.99, 10000)
    inlier_kp_left = [[point[0], point[1]] for point in key_points_left[inliers]]
    inlier_kp_right = [[point[0], point[1]] for point in key_points_right[inliers]]

    return inlier_kp_left, inlier_kp_right, homo
