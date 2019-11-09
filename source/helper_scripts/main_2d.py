import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

FRAMES_PATH = "../../data/datasets/frames2d/"
VIDEOS_PATH = "../../data/videos/"

if __name__ == "__main__":

    cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, "case-1-2D.avi"))

    timeInstants = [2000, 39500, 56000, 62500, 95000, 105500]
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

        cv2.imwrite(os.path.join(FRAMES_PATH, "frame-%d.png" % i), frames[i])
