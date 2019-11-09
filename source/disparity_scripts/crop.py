import cv2
import matplotlib.pyplot as plt


def crop():
    frame = cv2.imread('../../data/datasets/full3d/frame-0.png')

    mid = int(frame.shape[1] / 2)

    left = frame[:, 0:1750, :]
    right = frame[:, mid:mid+1750, :]

    ret2, thr_left = cv2.threshold(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                                   0,
                                   255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                   )

    ret3, thr_right = cv2.threshold(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                    )

    mask_left = cv2.findNonZero(thr_left)

    left_edge = mask_left[:, 0, 0].min()
    right_edge = mask_left[:, 0, 0].max()

    delta = right_edge - left_edge
    delta2 = 1440 - delta
    left_margin = int(delta2 / 2)
    right_margin = delta2 - left_margin

    plt.figure()
    plt.subplot(121)
    plt.imshow(left)
    plt.subplot(122)
    plt.imshow(left[:, left_edge - left_margin:right_edge + right_margin, :])

    mask_right = cv2.findNonZero(thr_right)

    left_edge = mask_right[:, 0, 0].min()
    right_edge = mask_right[:, 0, 0].max()

    delta = right_edge - left_edge
    delta2 = 1440 - delta
    left_margin = int(delta2 / 2)
    right_margin = delta2 - left_margin

    plt.figure()
    plt.subplot(121)
    plt.imshow(right)
    plt.subplot(122)
    plt.imshow(right[:, left_edge - left_margin:right_edge + right_margin, :])
    plt.show()


if __name__ == "__main__":
    crop()
