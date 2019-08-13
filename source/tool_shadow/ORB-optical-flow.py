from __future__ import print_function

import cv2 as cv
import numpy as np

# Lucas-Kanade algorithm's params
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


class App:
    def __init__(self, video_src):
        self.track_len = 10
        # re-computation interval
        self.detect_interval = 52
        # tracked points
        self.tracks = []
        # input video's fps
        self.fps = 26
        # OpenCV's video reader
        self.cam = cv.VideoCapture(video_src)
        # starting frame index
        self.frame_idx = 0
        # ORB key-points detector
        self.orb = cv.ORB_create(nfeatures=10)
        # centroid
        self.centroid = ()
        # set video frame rate
        self.cam.set(cv.CAP_PROP_FPS, self.fps)

    def run(self):
        l_edge, r_edge = 0, -1
        while True:
            # read next frame
            _ret, frame = self.cam.read()
            # crop frame to remove TrueVision logo which interferes ORB detection
            frame = frame[:1030, :, :]
            if self.frame_idx == 0:
                l_edge, r_edge = self.crop(frame)
            frame = frame[:, l_edge:r_edge, :]
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # create a working copy of the current frame
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                # reshape self.tracks
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # compute the optical flow for the current frame
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # compute the counter-proof optical flow for the previous frame
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                # check for good points, i.e. points found nearby each other in both frame and in both way
                dist = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = dist < 1
                new_tracks = []
                # list for points's coordinates
                xs, ys = [], []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    # store points' coordinated for centroid
                    xs.append(x)
                    ys.append(y)

                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (x, y), 3, (0, 0, 255), -1)
                # draw centroid
                if xs and ys:
                    self.centroid = (int(np.median(np.array(xs))), int(np.median(np.array(ys))))
                    cv.circle(vis, (self.centroid[0], self.centroid[1]), 4, (255, 0, 0), -1)
                self.tracks = new_tracks

                # cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 190, 0))
                draw_str(vis, (20, 20), 'track count: %d, frame: %d' % (len(self.tracks), self.frame_idx))

            # every 'self.detect_interval' frames compute ORB points
            if self.frame_idx % self.detect_interval == 0:
                # create mask
                # TODO: try to provide a smarter mask
                # mask = np.zeros(shape=(frame.shape[0], frame.shape[1]))
                mask = np.zeros_like(frame_gray)
                # mask = self.smart_mask(mask, frame_gray)
                mask[:, :] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                # detect ORB points
                kp = self.orb.detect(frame_gray, mask=mask)
                # sort points from the best to the worst one

                if kp is not None:
                    for p in kp:
                        # save detected points
                        self.tracks.append([(p.pt[0], p.pt[1])])

            # increment frame index
            self.frame_idx += 1
            self.prev_gray = frame_gray
            # show detected points
            cv.imshow('Lucas-Kanade track with ORB', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break

    def smart_mask(self, mask, frame):
        smart_mask = np.copy(mask)
        if self.frame_idx == 0:
            roi = cv.selectROI('Choose ROI', frame, fromCenter=False)
            smart_mask[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = 255
        else:
            smart_mask[:] = 255
        return smart_mask

    def crop(self, frame):
        _, thr = cv.threshold(cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
                              0,
                              255,
                              cv.THRESH_BINARY + cv.THRESH_OTSU
                              )
        crop_mask = cv.findNonZero(thr)

        left_edge = crop_mask[:, 0, 0].min()
        right_edge = crop_mask[:, 0, 0].max()

        return left_edge, right_edge


def main():
    try:
        video_src = "../../data/videos/clip-pat-2-video-2.mp4"
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
