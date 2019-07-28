from skimage import io
import cv2 as cv
import numpy as np


def ycbcr():
    img = cv.imread('../../data/datasets/2d_frames_folders/patient2-6/frame4937.png')
    res = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

    Cb = np.zeros(shape=img.shape, dtype=np.uint8)
    Cr = np.zeros(shape=img.shape, dtype=np.uint8)
    CrCb = np.zeros(shape=img.shape, dtype=np.uint8)

    Cb[:, :, 0] = res[:, :, 2]
    Cr[:, :, 2] = res[:, :, 1]
    CrCb[:, :, 0] = res[:, :, 2]
    CrCb[:, :, 2] = res[:, :, 1]

    cv.imshow('original', img)
    cv.waitKey()
    cv.imshow('Cb', Cb)
    cv.waitKey()
    cv.imshow('Cr', Cr)
    cv.waitKey()
    cv.imshow('CrCb', CrCb)
    cv.waitKey()


def glare_removal():
    img = io.imread('../../data/datasets/2d_frames_folders/patient2-2/frame4156.png')

    h, l, s = cv.split(cv.cvtColor(img, cv.COLOR_RGB2HLS))
    _, th = cv.threshold(l, 160, 255, cv.THRESH_BINARY)
    dst = cv.inpaint(img, th, 3, cv.INPAINT_TELEA)

    cv.imshow('original', cv.cvtColor(img, cv.COLOR_RGB2BGR))
    cv.waitKey()
    cv.imshow('glare removed', cv.cvtColor(dst, cv.COLOR_RGB2BGR))
    cv.waitKey()


def edges():

    img = io.imread('../../data/datasets/2d_frames_folders/patient2-6/frame4937.png')
    cv.imshow('original', cv.cvtColor(img, cv.COLOR_RGB2BGR))
    cv.waitKey()

    kernel = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

    '''kernel = np.array([[0, 0, 4],
                       [0, -4, 0],
                       [0, 0, 0]])'''

    # EDGES DETECTION ON RGB + DENOISING + CLOSING + ORB

    edges = cv.filter2D(img, -1, kernel)
    '''edges = cv.filter2D(edges, -1, np.array([[0, 1/4, 0],
                                            [1/4, 1, 1/4],
                                            [0, 1/4, 0]]))'''

    edgesclose = cv.fastNlMeansDenoising(edges, h=10, templateWindowSize=13, searchWindowSize=21)
    edgesclose = cv.morphologyEx(edgesclose, cv.MORPH_CLOSE, np.ones((8, 8), np.uint8))

    orb = cv.ORB_create(nfeatures=30)
    kp = orb.detect(edgesclose)
    kp, des = orb.compute(edgesclose, kp)
    tmp = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for p in kp:
        cv.circle(tmp, (int(p.pt[0]), int(p.pt[1])), 2, (0, 0, 255), -1)

    cv.imshow('edges', edges)
    cv.waitKey()
    cv.imshow('edgesclose', edgesclose)
    cv.waitKey()
    cv.imshow('points', tmp)
    cv.waitKey()

    # EDGES DETECTION CHANNEL R + ENHANCEMENT

    edgeR = cv.filter2D(img[:, :, 0], -1, kernel)
    edgeR = cv.filter2D(edgeR, -1, np.array([[0, 1/4, 0],
                                            [1/4, 1, 1/4],
                                            [0, 1/4, 0]]))

    orb = cv.ORB_create(nfeatures=30)
    kp = orb.detect(edgeR)
    kp, des = orb.compute(edgeR, kp)
    tmp = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for p in kp:
        cv.circle(tmp, (int(p.pt[0]), int(p.pt[1])), 2, (0, 0, 255), -1)

    cv.imshow('edgeR', edgeR)
    cv.waitKey()
    cv.imshow('points', tmp)
    cv.waitKey()

    # EDGES DETECTION CHANNEL G + ENHANCEMENT

    edgeG = cv.filter2D(img[:, :, 1], -1, kernel)
    edgeG = cv.filter2D(edgeG, -1, np.array([[0, 1/4, 0],
                                            [1/4, 1, 1/4],
                                            [0, 1/4, 0]]))

    cv.imshow('edgeG', edgeG)
    cv.waitKey()

    # EDGES DETECTION CHANNEL B + ENHANCEMENT

    edgeB = cv.filter2D(img[:, :, 2], -1, kernel)
    edgeB = cv.filter2D(edgeB, -1, np.array([[0, 1/4, 0],
                                            [1/4, 1, 1/4],
                                            [0, 1/4, 0]]))

    cv.imshow('edgeB', edgeB)
    cv.waitKey()


if __name__ == '__main__':
    ycbcr()
    #glare_removal()
    #edges()