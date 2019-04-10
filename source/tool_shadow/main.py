import os
import cv2
from matplotlib import pyplot as plt
from source.tool_shadow.utilities import hough_transform

from tqdm import tqdm

DATA_PATH = "../../data/datasets/frames2d/"
OUTPUT_PATH = "../../data/outputs/"

if __name__ == "__main__":
    image_name = "frame-6.png"

    imageNames = []
    for i in os.listdir(DATA_PATH):
        imageNames.append(i)

    id = 0
    for i in tqdm(imageNames):

        # load image
        img = cv2.imread(os.path.join(DATA_PATH, i))

        plt.figure()
        plt.title('image_rgb %d' % id)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        _, thr = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        plt.figure()
        plt.title('mask')
        plt.imshow(thr, 'gray')
        plt.show()

        plot_title = 'rgb_prob_B_%d' % id

        hough_transform(img,
                        img,
                        probabilistic=True,
                        save=False,
                        title=plot_title,
                        line_clr=(0, 255, 0)
                        )

        id += 1
