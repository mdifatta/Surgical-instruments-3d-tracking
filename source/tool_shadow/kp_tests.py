import numpy as np
from numpy import random
import cv2 as cv


def test_kp():

    img = cv.imread('../../data/datasets/2d_frames_folders/patient1-4/frame3069.png')

    cv.imshow('raw', img)
    cv.waitKey()

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

    ORB = cv.ORB_create()
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


if __name__ == '__main__':
    test_kp()
