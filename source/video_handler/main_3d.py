import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np

FRAMES_PATH = "../../data/datasets/"
VIDEOS_PATH = "../../data/videos/"

if __name__ == "__main__":
    cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, "case-1-3D.avi"))

    timeInstants = [1000, 12000, 52500, 79500, 108000]
    frames = []

    for t in timeInstants:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        _, f = cap.read()
        frames.append(f)

    for i in np.arange(len(timeInstants)):
        plt.figure()
        sec = timeInstants[i] / 1000.0
        plt.title("Sec %.1f" % sec)
        plt.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))

        plt.show()

        cv2.imwrite(os.path.join(FRAMES_PATH + 'full3d/', "full3d-%d.png" % i), frames[i])
        mid = int(frames[i].shape[1] / 2)
        cv2.imwrite(os.path.join(FRAMES_PATH + 'left3d/', "left3d-%d.png" % i), frames[i][:, 0:mid, :])
        cv2.imwrite(os.path.join(FRAMES_PATH + 'right3d/', "right3d-%d.png" % i), frames[i][:, mid:frames[i].shape[1], :])
