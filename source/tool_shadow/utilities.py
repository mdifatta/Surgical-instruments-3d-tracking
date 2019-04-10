import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from tqdm import tqdm
import os

OUTPUT_PATH = "./outputs/"


def hough_transform(image, blank, probabilistic=False, rho_acc=1, theta_acc=np.pi / 180, line_clr=(0, 0, 255),
                    threshold=142, threshold_p=70, min_line_length=0, max_line_gap=0,
                    save=False, title='Output'):

    img_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    v = np.median(img_g)
    sigma = .33

    if v > 191:
        lower = int(max(0, (1.0 - 2 * sigma) * (255 - v)))
        upper = int(min(85, (1.0 + 2 * sigma) * (255 - v)))
    elif v < 63:
        lower = int(max(0, (1.0 - 2 * sigma) * v))
        upper = int(min(85, (1.0 + 2 * sigma) * v))
    else:
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

    img_e = cv2.Canny(img_g, 50, 70, apertureSize=3)  # 80, 155, 3

    plt.figure()
    plt.title("Canny edges")
    plt.imshow(img_e, 'gray')

    # params
    # probabilistic = False
    # rho_acc = 1
    # theta_acc = np.pi / 180
    # red = (0, 0, 255)
    # threshold = 142
    # threshold_p = 100
    # min_line_length = 0
    # max_line_gap = 0

    if not probabilistic:
        lines = cv2.HoughLines(img_e, rho_acc, theta_acc, threshold, None, min_line_length, max_line_gap)
    else:
        lines = cv2.HoughLinesP(img_e, rho_acc, theta_acc, threshold_p, None, 0, 105)

    if lines is None:
        print("No lines found with these params")
    else:
        if not probabilistic:
            a, b, c = lines.shape
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv2.line(blank, pt1, pt2, line_clr, 2, cv2.LINE_AA)

            #lines_edges = cv2.addWeighted(image, 0.8, blank, 1, 0)

        else:
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(blank, (x1, y1), (x2, y2), line_clr, 2, cv2.LINE_AA)

        plt.figure()
        plt.title("Lines")
        plt.imshow(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
        if save:
            cv2.imwrite(os.path.join(OUTPUT_PATH, title+'.png'), blank)

    plt.show()


def enhance_contrast_brightness(image, alpha=1.3, beta=20):
    print('enhancing contrast and brightness...')
    for i in tqdm(np.arange(image.shape[0])):
        for j in np.arange(image.shape[1]):
            for c in np.arange(0, 2):
                image[i, j, c] = alpha * image[i, j, c] + beta
                np.clip(image[i, j, c], 0, 255)
    return image


def sharpening(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def downscaling(image):
    return cv2.pyrDown(image)


def tune_channel(image, red_coeff=1, green_coeff=1, blue_coeff=1):
    image[:, :, 0] = image[:, :, 0] * blue_coeff  # blue channel
    image[:, :, 1] = image[:, :, 1] * green_coeff  # green channel
    image[:, :, 2] = image[:, :, 2] * red_coeff  # red channel
    return image
