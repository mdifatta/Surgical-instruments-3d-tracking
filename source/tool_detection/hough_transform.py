import cv2
from matplotlib import pyplot as plt

from source.tool_detection.utilities import hough_transform

DATA_PATH = "../../data/videos_2D/"
OUTPUT_PATH = "../../data/outputs/"

if __name__ == "__main__":
    filename = "case-3-2D.avi"

    cam = cv2.VideoCapture(DATA_PATH + filename)
    cam.set(cv2.CAP_PROP_POS_FRAMES, 2000)

    while cam.isOpened():
        _ret, full_frame = cam.read()

        plt.figure()
        plt.title('image_rgb')
        plt.imshow(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))

        _, thr = cv2.threshold(cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.figure()
        plt.title('mask')
        plt.imshow(thr, 'gray')
        plt.show()

        plot_title = 'rgb_prob_B'

        hough_transform(full_frame,
                        full_frame,
                        probabilistic=True,
                        save=False,
                        title=plot_title,
                        line_clr=(0, 255, 0)
                        )
