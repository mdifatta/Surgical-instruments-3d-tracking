import cv2
import numpy as np
from tqdm import tqdm


def main():
    frames_step = 10
    video_name = "clip-6-3D.mp4"
    folder_name = "clip-6"

    cap = cv2.VideoCapture('../../data/videos/%s' % video_name)

    id = 13000
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = int(length / 10) * 10

    for i in tqdm(np.arange(start=0, step=frames_step, stop=length)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            frame = frame[:, :1920, :]
            cv2.imwrite('../../data/datasets/2d_frames_folders/%s/frame%d.png' % (folder_name, id),
                        frame, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            id = id + 1
        else:
            print('exiting...')
            break

    cap.release()


if __name__ == '__main__':
    main()
