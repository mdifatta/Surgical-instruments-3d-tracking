from __future__ import print_function

import numpy as np
import cv2 as cv


lk_params = dict(winSize =(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=5,
                      qualityLevel=0.3,
                      minDistance=2,
                      blockSize=10)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 60
        self.tracks = []
        self.fps = 30
        self.cam = cv.VideoCapture(video_src)
        self.cam.set(cv.CAP_PROP_FPS, self.fps)
        self.frame_idx = 0

    def run(self):

        while True:
            _ret, frame = self.cam.read()
            frame = frame[:1000, 450:1500, :]
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                dist = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = dist < 1
                new_tracks = []
                xs, ys = [], []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))

                    xs.append(x)
                    ys.append(y)

                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 3, (0, 0, 255), -1)
                if xs and ys:
                    cv.circle(vis, (int(np.median(np.array(xs))), int(np.median(np.array(ys)))), 4, (255, 0, 0), -1)
                self.tracks = new_tracks

                # cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 190, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break


def main():
    try:
        video_src = "../../data/videos/clip-pat-1-video-4.mp4"
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()