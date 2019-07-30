import cv2 as cv
import numpy as np
from numpy import random


def test_kp():

    #img = cv.imread('../../data/datasets/2d_frames_folders/patient2-2/frame4808.png')
    cam = cv.VideoCapture("../../data/videos/clip-pat-2-video-2.mp4")
    cam.set(cv.CAP_PROP_FPS, 30)
    cam.set(cv.CAP_PROP_POS_FRAMES, 1260)
    _, img = cam.read()

    cv.imshow('raw', img)
    cv.waitKey()

    img = crop(img[:1030, :, :])

    mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    #mask[100:150, 120:160] = 255
    mask[:, :] = 255

    # GOOD FEATURES TO TRACK

    feature_params = dict(maxCorners=5,
                          qualityLevel=0.3,
                          minDistance=2,
                          blockSize=10)
    gftt = np.copy(img)
    p = cv.goodFeaturesToTrack(cv.cvtColor(img, cv.COLOR_BGR2GRAY), mask=mask, **feature_params)

    for x, y in np.float32(p).reshape(-1, 2):
        cv.circle(gftt, (x, y), 2, (random.randint(0, 256), 255, random.randint(0, 256)), 1, cv.LINE_8)

    cv.imshow('gftt - Points: %d' % len(p), gftt)
    cv.waitKey()

    # ORB

    orb = np.copy(img)

    ORB = cv.ORB_create(nfeatures=20)
    kp = ORB.detect(orb, mask=mask)
    kp, des = ORB.compute(orb, kp)

    kp.sort(key=lambda x: x.response, reverse=True)

    for p in kp[:]:
        cv.circle(orb, (int(p.pt[0]), int(p.pt[1])), 2,
                  (random.randint(0, 200), random.randint(0, 256), random.randint(0, 256)), 1, cv.LINE_8)

    cv.imshow('orb - n=%d - Showing best 5 points' % len(kp), orb)
    cv.waitKey()

    # SURF

    surf = np.copy(img)
    SURF = cv.xfeatures2d.SURF_create()
    kp = SURF.detect(surf, mask=mask)
    kp, des = SURF.compute(surf, kp)

    kp.sort(key=lambda x: x.response, reverse=True)

    for p in kp[:]:
        cv.circle(surf, (int(p.pt[0]), int(p.pt[1])), 2, (random.randint(0, 256), 255, random.randint(0, 256)), 1,
                  cv.LINE_8)

    cv.imshow('surf - n=%d - Showing best 5 points' % len(kp), surf)
    cv.waitKey()

    # SIFT

    sift = np.copy(img)
    SIFT = cv.xfeatures2d.SIFT_create()
    kp = SIFT.detect(sift, mask=mask)
    kp, des = SIFT.compute(sift, kp)

    kp.sort(key=lambda x: x.response, reverse=True)

    for p in kp[:]:
        cv.circle(sift, (int(p.pt[0]), int(p.pt[1])), 2, (random.randint(0, 256), 255, random.randint(0, 256)), 1,
                  cv.LINE_8)

    cv.imshow('sift - n=%d - Showing best 5 points' % len(kp), sift)
    cv.waitKey()


def crop(frame):
    _, thr = cv.threshold(cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
                          0,
                          255,
                          cv.THRESH_BINARY + cv.THRESH_OTSU
                         )
    crop_mask = cv.findNonZero(thr)

    left_edge = crop_mask[:, 0, 0].min()
    right_edge = crop_mask[:, 0, 0].max()

    return frame[:, left_edge:right_edge, :]


if __name__ == '__main__':
    test_kp()
