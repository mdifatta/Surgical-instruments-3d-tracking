import cv2
import numpy as np
from tqdm import tqdm
#                          ##################################################
#                          #########   TO BE RUN ON GOOGLE COLAB   ##########
#                          ##################################################


def main():
    frames_step = 6
    video_name = "case-1-2D.avi"
    folder_name = "video-1"

    cap = cv2.VideoCapture('../%s' % video_name)

    id = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = int(length / 10) * 10

    for i in tqdm(np.arange(start=0, step=frames_step, stop=length)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite('../%s/frame%d.png' % (folder_name, id), frame, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            id = id + 1
        else:
            print('exiting...')
            break

    cap.release()


if __name__ == '__main__':
    main()
