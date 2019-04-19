import cv2
import os
from tqdm import tqdm
#                          ##################################################
#                          #########   TO BE RUN ON GOOGLE COLAB   ##########
#                          ##################################################


def crop_and_resize(path):
    crop_shape = (1280, 960)

    frame = cv2.imread(path)[:crop_shape[1], :, :]

    # crop
    ret, thr = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             0,
                             255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU
                             )

    mask = cv2.findNonZero(thr)

    left_edge = mask[:, 0, 0].min()
    right_edge = mask[:, 0, 0].max()

    width = crop_shape[0]

    delta = right_edge - left_edge
    delta2 = width - delta
    left_margin = int(delta2 / 2)
    right_margin = delta2 - left_margin

    frame = frame[:, left_edge - left_margin:right_edge + right_margin, :]

    # resize
    frame = cv2.resize(frame, dsize=(0, 0), fx=.5, fy=.5)

    # overwrite
    cv2.imwrite(path, frame)


if __name__ == "__main__":
    folder_names = os.listdir('../data/')
    for f in folder_names:
        frame_names = os.listdir('../data/' + f)
        for fr in tqdm(frame_names):
            crop_and_resize('../data/' + f + '/' + fr)
