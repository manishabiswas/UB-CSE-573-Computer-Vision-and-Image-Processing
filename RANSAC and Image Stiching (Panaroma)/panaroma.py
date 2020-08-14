"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random
import math

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    'Read the two images in grayscales'
    left_gray_image = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray_image = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    'Computing SIFT features'
    sift = cv2.xfeatures2d.SIFT_create()
    keypoint_left, descriptor_left = sift.detectAndCompute(left_gray_image, None)
    keypoint_right, descriptor_right = sift.detectAndCompute(right_gray_image, None)

    FLANN_INDEX_KDTREE = 0

    check_index_params = dict(algorithm=FLANN_INDEX_KDTREE, tress=5)
    check_search_params = dict(checks=100)

    '''Create object for FLANN matching'''
    doFlann = cv2.FlannBasedMatcher(check_index_params, check_search_params)
    matches = doFlann.knnMatch(descriptor_left, descriptor_right, k=2)

    Keypoints = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            Keypoints.append(m)

    MIN_NUM_MATCHES = 10

    if len(Keypoints) > MIN_NUM_MATCHES:
        coord1 = np.array([keypoint_left[m.queryIdx].pt for m in Keypoints])
        np.reshape(coord1, (-1, 1, 2))
        coord1 = np.float32(coord1)
        coord2 = np.array([keypoint_right[m.trainIdx].pt for m in Keypoints])
        np.reshape(coord2, (-1, 1, 2))
        coord2 = np.float32(coord2)

        H, homographyStatus = cv2.findHomography(coord1, coord2, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - %d/%d" % (len(Keypoints), MIN_NUM_MATCHES))

    'Read the shapes of the two images'
    left_h, left_w = left_gray_image.shape
    right_h, right_w = right_gray_image.shape

    'Save the corners of each image in a form of list of list'
    points_m1 = np.float32([[0, 0], [0, left_h], [left_w, left_h], [left_w, 0]]).reshape(-1, 1, 2)
    points_m2 = np.float32([[0, 0], [0, right_h], [right_w, right_h], [right_w, 0]]).reshape(-1, 1, 2)

    'Compute the translation and rotation of each corner of left image w.r.t the right image using Homography matrix'
    points_m1_modify = cv2.perspectiveTransform(points_m1, H)

    '''Add the translated corners of the left image to the corners of the right image in the form of a 
    list of list for each corner location on the canvas. We add alon axis=0, i.e. along the rows'''
    all_mPoints = np.concatenate((points_m2, points_m1_modify), axis=0)

    '''Find the minimum and maximum values from all the corners to create the size of the frame at a 
     distance of +-1 '''
    [pano_xmin, pano_ymin] = np.int32(all_mPoints.min(axis=0).ravel() - 1.0)
    [pano_xmax, pano_ymax] = np.int32(all_mPoints.max(axis=0).ravel() + 1.0)

    'Calculate the left upper most and lower most corners of the frame'
    transformationM = [-pano_ymin, -pano_xmin]

    'Compute the Translated matrix for the new frame, w.r.t which our H will be translated'
    translatedH = np.array([[1, 0, transformationM[1]], [0, 1, transformationM[0]], [0, 0, 1]])

    'warp the left image w.r.t the right image'
    img_pano = cv2.warpPerspective(left_img, translatedH.dot(H), (pano_xmax - pano_xmin, pano_ymax - pano_ymin))
    img_pano[transformationM[0]:right_h + transformationM[0],transformationM[1]:right_w + transformationM[1]] = right_img

    return img_pano
# raise NotImplementedError

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg', result_image)
