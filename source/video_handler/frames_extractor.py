import cv2
import numpy as np
from tqdm import tqdm


def main():
    cap = cv2.VideoCapture('../../data/videos/case-1-2D.avi')
    id = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    length = int(length / 10) * 10

    for i in tqdm(np.arange(start=0, step=3, stop=length)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite('../../data/datasets/frames2d/frame%d.png' % id, frame, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            id = id + 1
        else:
            print('exiting...')
            break

    cap.release()


if __name__ == '__main__':
    main()
