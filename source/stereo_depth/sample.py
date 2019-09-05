import cv2 as cv
import numpy as np
from scipy import signal


def crop(frame):
    _, thr = cv.threshold(cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
                          0,
                          255,
                          cv.THRESH_BINARY + cv.THRESH_OTSU
                          )
    crop_mask = cv.findNonZero(thr)

    left_edge = crop_mask[:, 0, 0].min()
    right_edge = crop_mask[:, 0, 0].max()

    return left_edge, right_edge


def sharpening(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    return cv.filter2D(img, -1, kernel)


def CLAHE(img):
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def gaussian_weights(kernlen, std=2.5):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def tip_averaging(tip):
    # Gaussian weights for the tip
    weights = gaussian_weights(kernlen=100)

    weights[tip <= 20] = 0
    # weighted average disparity value for the tip
    avg_tip = np.average(tip, weights=weights)
    return avg_tip


def retina_averaging(retina, tip_avg):
    # clip retina's disparities
    retina = np.clip(retina, 1e-7, int(tip_avg))

    # weights for the background
    weights = np.full_like(retina, fill_value=tip_avg, dtype=np.uint8)
    weights = np.clip(np.subtract(weights, retina), a_min=1e-7, a_max=int(tip_avg))
    # get rid of meaningless areas
    weights[retina <= 10] = 1e-7
    # get rid of white noise near edges and outliers
    weights[retina > tip_avg] = 1e-7
    # normalize weights
    weights = cv.normalize(weights, None, 1e-7, 1, cv.NORM_MINMAX, cv.CV_32F)

    cv.imshow('Retina w', weights)
    # weighted average and median disparity value for the retina
    avg_retina = np.average(retina, weights=weights)
    return avg_retina


def compare_disparities(disparity_map, centroid, tip_mask_size=100, retina_mask_size=320):
    # TODO: dynamic size of the masks

    # filter masks from the disparity map
    tip = disparity_map[centroid[1] - (tip_mask_size // 2):centroid[1] + (tip_mask_size // 2),
                        centroid[0] - (tip_mask_size//2):centroid[0] + (tip_mask_size//2)]
    retina = disparity_map[centroid[1] - (retina_mask_size // 2):centroid[1] + (retina_mask_size // 2),
                           centroid[0] - (retina_mask_size//2):centroid[0] + (retina_mask_size//2)]

    # compute average disparity for the tool's tip
    tip_avg_disparity = tip_averaging(tip)
    # compute average disparity for the background retina
    retina_avg_disparity = retina_averaging(retina, tip_avg_disparity)

    return tip_avg_disparity - retina_avg_disparity


class StereoParams:
    # ################## sets of params for good disparities ##################
    params_no_sharp_bgr = dict(minDisparity=-6,
                               numDisparities=64,
                               blockSize=9,
                               speckleRange=7,
                               speckleWindowSize=180,
                               disp12MaxDiff=13,
                               mode=cv.STEREO_SGBM_MODE_HH
                               )

    params_no_sharp_bgr_P1_P2 = dict(minDisparity=-7,
                                     numDisparities=112,
                                     blockSize=7,  # or 9
                                     speckleRange=15,
                                     speckleWindowSize=200,
                                     disp12MaxDiff=40,
                                     # mode=cv.STEREO_SGBM_MODE_HH,
                                     P1=250,
                                     P2=5000  # or 7000
                                     )

    params_clahe_bgr_P1_P2 = dict(minDisparity=-7,
                                  numDisparities=112,
                                  blockSize=5,
                                  speckleRange=8,
                                  speckleWindowSize=200,
                                  disp12MaxDiff=5,
                                  P1=1,
                                  P2=4000,
                                  uniquenessRatio=4
                                  )

    params_clahe_bgr_P1_P2_v2 = dict(
                                     minDisparity=-4,
                                     numDisparities=128,
                                     blockSize=3,
                                     speckleRange=5,
                                     speckleWindowSize=150,
                                     disp12MaxDiff=7,
                                     P1=20,
                                     P2=5000,
                                     uniquenessRatio=7
                                    )

    params_no_sharp_gray = dict(minDisparity=5,
                                numDisparities=80,
                                blockSize=13,
                                speckleRange=7,
                                speckleWindowSize=130,
                                disp12MaxDiff=50,
                                mode=cv.STEREO_SGBM_MODE_HH
                                )

    params_sharp_bgr = dict(minDisparity=-16,  # try to increase over 0
                            numDisparities=80,
                            blockSize=7,
                            speckleRange=6,
                            speckleWindowSize=130,
                            disp12MaxDiff=25,  # try to increase over 30/40
                            mode=cv.STEREO_SGBM_MODE_HH
                            )

    params_sharp_bgr_P1_P2 = dict(minDisparity=-4,
                                  numDisparities=112,
                                  blockSize=5,
                                  speckleRange=10,
                                  speckleWindowSize=200,
                                  disp12MaxDiff=25,
                                  P1=250,
                                  P2=2000
                                  )


def depth(left, right, centroid):

    # create stereo matcher object
    stereo = cv.StereoSGBM_create(
        **StereoParams.params_clahe_bgr_P1_P2_v2
        )

    # match left and right frames
    disparity = stereo.compute(
        CLAHE(left),
        CLAHE(right)
    )

    # values from matching are float, normalize them between 0-255 as integer
    disparity = cv.normalize(cv.resize(disparity, (0, 0), fx=1/.6, fy=1/.6), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    cv.imshow('Disp', disparity)

    delta_disparity = compare_disparities(disparity, centroid)
    print(delta_disparity)
    cv.waitKey()

    # (time.time() * 1000) - start


if __name__ == '__main__':
    frames_ids = [0, 1, 2, 3, 4]
    tool_tips = [(556, 638), (465, 540), (285, 345), (250, 175), (264, 364)]
    for f, t in zip(frames_ids, tool_tips):
        img = cv.imread('../../data/datasets/left3d/left3d-%d.png' % f)
        left_e, right_e = crop(img[:1030, :, :])
        left_img = img[:1030, left_e:right_e, :]
        right_img = cv.imread('../../data/datasets/right3d/right3d-%d.png' % f)[:1030, left_e:right_e, :]
        imgL = left_img
        imgR = right_img
        depth(imgL, imgR, t)
