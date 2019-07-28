import cv2 as cv
import numpy as np

class App:
    def __init__(self, video_src):
        self.fps = 30
        self.cam = cv.VideoCapture(video_src)
        self.cam.set(cv.CAP_PROP_FPS, self.fps)
        self.orb = cv.ORB_create()

    def run(self):
        _, current_frame = self.cam.read()
        current_frame = cv.cvtColor(current_frame[:1000, 450:1500, :], cv.COLOR_BGR2GRAY)
        nxt, next_frame = self.cam.read()
        next_frame = cv.cvtColor(next_frame[:1000, 450:1500, :], cv.COLOR_BGR2GRAY)

        mask = np.zeros(shape=(current_frame.shape[0], current_frame.shape[1]), dtype=np.uint8)
        mask[700:850, 650:800] = 255

        while nxt:
            kp1 = self.orb.detect(current_frame, mask=mask)
            kp1, des1 = self.orb.compute(current_frame, kp1)

            kp2 = self.orb.detect(next_frame, mask)
            kp2, des2 = self.orb.compute(next_frame, kp2)

            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)

            matches = sorted(matches, key=lambda x: x.distance)

            res = cv.drawMatches(current_frame, kp1, next_frame, kp2, matches[:10], None, flags=0)

            cv.imshow('t vs t+1 frame', res)
            # cv.waitKey()
            ch = cv.waitKey(1)
            if ch == 27:
                break

            current_frame = next_frame
            nxt, next_frame = self.cam.read()
            next_frame = cv.cvtColor(next_frame[:1000, 450:1500, :], cv.COLOR_BGR2GRAY)


def main():
    try:
        video_src = "../../data/videos/clip-pat-1-video-4.mp4"
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    main()