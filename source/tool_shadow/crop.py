import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def crop(path):
    frame = cv2.imread(path)[:1000, :, :]

    ret, thr = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             0,
                             255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU
                             )

    mask = cv2.findNonZero(thr)

    left_edge = mask[:, 0, 0].min()
    right_edge = mask[:, 0, 0].max()

    delta = right_edge - left_edge
    delta2 = frame.shape[0] - delta
    left_margin = int(delta2 / 2)
    right_margin = delta2 - left_margin

    frame = cv2.resize(frame[:, left_edge - left_margin:right_edge + right_margin, :], dsize=(512, 512))

    cv2.imwrite(path, frame)

    '''
    plt.figure()
    plt.subplot(121)
    plt.imshow(frame)
    plt.subplot(122)
    plt.imshow(cv2.resize(frame[:, left_edge - left_margin:right_edge + right_margin, :], dsize=(512, 512)))
    plt.show()
    '''


if __name__ == "__main__":
    #folder_names = os.listdir('../../data/datasets/distance_frames_folders/')
    #for f in folder_names:
        f = 'prova'
        frame_names = os.listdir('../../data/datasets/distance_frames_folders/' + f)
        for fr in tqdm(frame_names):
            crop('../../data/datasets/distance_frames_folders/' + f + '/' + fr)
