import cv2
import numpy as np
from tqdm import tqdm
#                          ##################################################
#                          #########   TO BE RUN ON GOOGLE COLAB   ##########
#                          ##################################################


def main():
    cap = cv2.VideoCapture('../case-1-2D.avi')
    # TODO: the index must be consecutive across videos so first generate frames then
    # rename frames with consecutive filenames
    id = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    length = int(length / 10) * 10

    frames_step = 6

    for i in tqdm(np.arange(start=0, step=frames_step, stop=length)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite('../case-1/frame%d.png' % id, frame, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            id = id + 1
        else:
            print('exiting...')
            break

    cap.release()


if __name__ == '__main__':
    main()
