import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

DATA_PATH = "./data/"
OUTPUT_PATH = "./outputs/"


def matching_points(left_img, right_img, save=False, filename='output'):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_img, None)
    kp2, des2 = orb.detectAndCompute(right_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(left_img, kp1, right_img, kp2, matches, None, flags=2)

    plt.figure()
    plt.title("Matching points")
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

    if save:
        cv2.imwrite(os.path.join(OUTPUT_PATH, filename+'.png'), img3)


def enhance_contrast_brightness(left_img, right_img):
    # contrast and brightness params respectively
    alpha = 1.3
    beta = 20
    print('enhancing contrast and brightness...')
    for i in tqdm(np.arange(left_img.shape[0])):
        for j in np.arange(left_img.shape[1]):
            for c in np.arange(0, 2):
                left_img[i, j, c] = alpha * left_img[i, j, c] + beta
                np.clip(left_img[i, j, c], 0, 255)
                right_img[i, j, c] = alpha * right_img[i, j, c] + beta
                np.clip(right_img[i, j, c], 0, 255)
    return left_img, right_img


def downscaling(left_img, right_img):
    print('downscaling images...')
    left_img = cv2.pyrDown(left_img)
    right_img = cv2.pyrDown(right_img)
    return left_img, right_img


def normalize(disp_map):
    return cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def disp_wls_smoother(left_matcher, left_img, right_img):
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    left_disp = left_matcher.compute(left_img, right_img)
    right_disp = right_matcher.compute(right_img, left_img)
    return wls_filter.filter(left_disp, left_img, disparity_map_right=right_disp)


def circle_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, gray = cv2.threshold(gray, 2, 255, cv2.CV_8UC1)

    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]

    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               4,
                               rows / 8,
                               param1=80,
                               param2=100,
                               minRadius=500,
                               maxRadius=700
                               )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 0, 255), 3)
            # circle outline
            radius = i[2]
            cv2.circle(image, center, radius, (0, 255, 0), 3)
    else:
        print("No circles detected.")

    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(gray, 'gray')
    plt.show()
