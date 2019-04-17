import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from source.stereo_depth.utilities import matching_points, downscaling, normalize, disp_wls_smoother, circle_hough

DATA_PATH = "../../data/datasets/"
OUTPUT_PATH = "../../data/outputs/"
left_name = "left3d-1.png"
right_name = "right3d-1.png"

if __name__ == "__main__":

    left_names = ["left3d-0.png", "left3d-1.png", "left3d-2.png", "left3d-3.png", "left3d-4.png"]
    right_names = ["right3d-0.png", "right3d-1.png", "right3d-2.png", "right3d-3.png", "right3d-4.png"]

    for i in np.arange(len(left_names)):

        print("loading stereo images ...")
        left_img = cv2.imread(os.path.join(DATA_PATH + 'left3d', left_names[i]))
        right_img = cv2.imread(os.path.join(DATA_PATH + 'right3d', right_names[i]))

        # circle_hough(left_img) used to find ROI

        left_img, right_img = downscaling(left_img, right_img)

        # stereo SGBM params
        minDisp = -15
        numDisp = 96
        blockSize = 1
        speckleRange = 2
        speckleWindowSize = 4

        stereo = cv2.StereoSGBM_create(minDisparity=minDisp,
                                       numDisparities=numDisp,
                                       blockSize=blockSize,
                                       speckleRange=speckleRange,
                                       speckleWindowSize=speckleWindowSize
                                       )

        print("computing disparities ...")
        disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

        print("normalizing ...")
        disparity = normalize(disparity)

        print("showing results ...")

        plt.figure()
        plt.suptitle("original images")
        plt.subplot(121).set_title("left3d")
        plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        plt.subplot(122).set_title("right3d")
        plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))

        plt.figure()
        plt.title("disparities map")
        plt.imshow(disparity, 'inferno')
        plt.colorbar(orientation='horizontal')
        cv2.imwrite(os.path.join(OUTPUT_PATH, 'disparity_%d.png' % i), disparity)

        smoothed = disp_wls_smoother(stereo, left_img, right_img)
        plt.figure()
        plt.title("disparities map")
        plt.imshow(smoothed, 'inferno')
        plt.colorbar(orientation='horizontal')
        cv2.imwrite(os.path.join(OUTPUT_PATH, 'disparity_smooth_%d.png' % i), smoothed)

        plt.show()

        match_title = 'matching_points_%d' % i
        matching_points(left_img, right_img,
                        save=True,
                        filename=match_title)
