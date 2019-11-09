import cv2 as cv
import numpy as np
from skimage import io

frames = ['/case-1/frame1.png', '/case-1/frame98.png',
          '/case-1/frame402.png', '/case-1/frame649.png', '/case-1/frame966.png',
          '/case-2/frame1253.png', '/case-2/frame1293.png', '/case-2/frame1663.png',
          '/case-3/frame1876.png', '/case-3/frame1709.png', '/case-10/frame1886.png',
          '/case-10/frame3446.png', '/case-10/frame2873.png', '/case-10/frame2117.png',
          '/case-11/frame3584.png', '/case-11/frame3811.png', '/case-12/frame3871.png',
          '/case-12/frame4268.png', '/case-13/frame4841.png', '/case-13/frame4937.png']


def ycbcr(frame):
    img = cv.imread('../../data/datasets/2d_frames_folders' + frame)
    res = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

    Cb = np.zeros(shape=img.shape, dtype=np.uint8)
    Cr = np.zeros(shape=img.shape, dtype=np.uint8)
    CrCb = np.zeros(shape=img.shape, dtype=np.uint8)

    Cb[:, :, 0] = res[:, :, 2]
    Cr[:, :, 2] = res[:, :, 1]
    CrCb[:, :, 0] = res[:, :, 2]
    CrCb[:, :, 2] = res[:, :, 1]

    mask = np.zeros(shape=(Cr.shape[0], Cr.shape[1]))
    for i in range(Cr.shape[0]):
        for j in range(Cr.shape[1]):
            if Cr[i, j, 2] > 130 and Cr[i, j, 2] <= 150:
                mask[i, j] = 255
                # TODO: provare questo approccio su immagine pre-processata con CLAHE

    cv.imshow('original', img)
    cv.waitKey()
    cv.imshow('Cb', Cb)
    cv.waitKey()
    cv.imshow('Cr', Cr)
    cv.waitKey()
    cv.imshow('CrCb', CrCb)
    cv.waitKey()
    cv.imshow('Range', mask)
    cv.waitKey()

    
def shadow_enhance(frame):
    img = cv.imread('../../data/datasets/2d_frames_folders' + frame, 1)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(25, 25))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    cv.imshow('raw', img)
    cv.imshow('clahe+sharp', edges)

    cv.waitKey()


def shadow(frame):
    img = cv.imread('../../data/datasets/2d_frames_folders' + frame, 1)
    cv.imshow("img", img)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv.split(lab)

    cl = clahe.apply(l)
    cv.imshow('clahe l', cl)
    res1 = cv.merge((cl, a, b))
    res1 = cv.cvtColor(res1, cv.COLOR_LAB2BGR)
    cv.imshow('clahe color', res1)

    mog2 = cv.createBackgroundSubtractorKNN(detectShadows=True)
    print(cv.BackgroundSubtractorKNN.getShadowThreshold(mog2))
    res2 = mog2.apply(img)
    cv.imshow('mog', res2)

    cv.waitKey()


def glare_removal():
    img = io.imread('../../data/datasets/2d_frames_folders/case-12/frame4156.png')

    h, l, s = cv.split(cv.cvtColor(img, cv.COLOR_RGB2HLS))
    _, th = cv.threshold(l, 160, 255, cv.THRESH_BINARY)
    dst = cv.inpaint(img, th, 3, cv.INPAINT_TELEA)

    cv.imshow('original', cv.cvtColor(img, cv.COLOR_RGB2BGR))
    cv.waitKey()
    cv.imshow('glare removed', cv.cvtColor(dst, cv.COLOR_RGB2BGR))
    cv.waitKey()


def edges(frame):

    img = io.imread('../../data/datasets/2d_frames_folders' + frame)
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
    for f in frames:
        ycbcr(f)
        #glare_removal()
        #edges(f)
        #shadow_enhance(f)
        #shadow(f)