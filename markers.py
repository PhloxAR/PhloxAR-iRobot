# -*- coding: utf-8 -*-

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import cv2
import numpy as np


def _order_points(points):
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    ordered_points = np.zeros((4, 2), dtype='float32')

    ordered_points[0] = points[np.argmin(s)]
    ordered_points[2] = points[np.argmax(s)]
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[3] = points[np.argmin(diff)]

    return ordered_points


def _max_width_height(points):
    tl, tr, br, bl = points

    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    bot_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(top_width), int(bot_width))

    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))

    return max_width, max_height


def _topdown_points(max_width, max_height):
    return np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype='float32')


def get_topdown_quad(image, src):
    # src and dst points
    src = _order_points(src)

    (max_width, max_height) = _max_width_height(src)
    dst = _topdown_points(max_width, max_height)

    # warp perspective
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, _max_width_height(src))

    return warped


def get_marker_pattern(image, black_threshold, white_threshold):
    # collect pixel from each cell (left to right, top to bottom)
    cells = []

    cell_half_width = int(round(image.shape[1] / 10.0))
    cell_half_height = int(round(image.shape[0] / 10.0))

    row1 = cell_half_height * 3
    row2 = cell_half_height * 5
    row3 = cell_half_height * 7
    col1 = cell_half_width * 3
    col2 = cell_half_width * 5
    col3 = cell_half_width * 7

    cells.append(image[row1, col1])
    cells.append(image[row1, col2])
    cells.append(image[row1, col3])
    cells.append(image[row2, col1])
    cells.append(image[row2, col2])
    cells.append(image[row2, col3])
    cells.append(image[row3, col1])
    cells.append(image[row3, col2])
    cells.append(image[row3, col3])

    # threshold pixels to either black or white
    for idx, val in enumerate(cells):
        if val < black_threshold:
            cells[idx] = 0
        elif val > white_threshold:
            cells[idx] = 1
        else:
            return None

    return cells


def get_vectors(image, points, mtx, dist):
    # order points
    points = _order_points(points)

    # set up criteria, image, points and axis
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgp = np.array(points, dtype='float32')

    objp = np.array([[0., 0., 0.], [1., 0., 0.],
                     [1., 1., 0.], [0., 1., 0.]], dtype='float32')

    # calculate rotation and translation vectors
    cv2.cornerSubPix(gray, imgp, (11, 11), (-1, -1), criteria)
    # backup
    # rvecs, tvecs, _ = cv2.solvePnPRansac(objp, imgp, mtx, dist)
    ret, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

    return rvecs, tvecs


class Marker(object):
    QUADRILATERAL_POINTS = 4
    BLACK_THRESHOLD = 100
    WHITE_THRESHOLD = 155
    MARKER_NAME_INDEX = 3

    def __init__(self):
        with np.load('calibration/mine_cam.npz') as camfile:
            self.mtx, self.dist, _, _ = [
                camfile[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')
            ]

    def detect(self, image):
        markers = []

        # Stage 1: Detect edges in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 100, 200)

        # Stage 2: Find contours
        _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        cv2.drawContours(image, contours, -1, (0, 255, 255), 3)

        for contour in contours:
            # Stage 3: Shape check
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01*perimeter, True)

            if len(approx) == self.QUADRILATERAL_POINTS:
                # Stage 4: Perspective warping
                topdown_quad = get_topdown_quad(gray, approx.reshape(4, 2))

                # Stage 5: Border check
                if topdown_quad[(topdown_quad.shape[0] / 100.0) * 5,
                                (topdown_quad.shape[1] / 100.0) * 5] > self.BLACK_THRESHOLD:
                    continue

                # Stage 6: Get marker pattern
                marker_pattern = None

                try:
                    marker_pattern = get_marker_pattern(topdown_quad,
                                                        self.BLACK_THRESHOLD,
                                                        self.WHITE_THRESHOLD)
                except:
                    continue

                if not marker_pattern:
                    continue

                # Stage 7: Match marker pattern
                found, rotation, name = match_marker_pattern(marker_pattern)

                if found:
                    # Stage 8: Duplicate marker check
                    if name in [marker[self.MARKER_NAME_INDEX] for marker in markers]
                        continue

                    # Stage 9: Get rotation and translation vectors
                    rvecs, tvecs = get_vectors(image, approx.reshape(4, 2),
                                               self.mtx, self.dist)
                    markers.append([rvecs, tvecs, rotation, name])

        return markers
