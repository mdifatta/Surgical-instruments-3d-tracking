from __future__ import print_function

import cv2 as cv
import numpy as np

# Lucas-Kanade algorithm's params
lk_params = dict(winSize=(15, 15),
                 maxLevel=7,
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
        self.centroid = None
        # Centroid movement vector
        self.centroid_mov_vector = None
        # set video frame rate
        self.cam.set(cv.CAP_PROP_FPS, self.fps)
        # frames count
        self.frames_count = self.cam.get(cv.CAP_PROP_FRAME_COUNT)

        self.reset_mask = True

    def run(self):
        l_edge, r_edge = 0, -1

        if not self.cam.isOpened():
            print('Error reading video')
            return

        while self.cam.isOpened():  #self.frame_idx < self.frames_count:
            # read next frame
            _ret, frame = self.cam.read()
            if not _ret:
                self.frame_idx += 1
                continue
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
                    cv.circle(vis, (x, y), 5, (0, 255, 0), -1)
                # draw centroid
                if xs and ys:
                    # compute new centroid coordinates
                    new_centroid_x = int(np.median(np.array(xs)))
                    new_centroid_y = int(np.median(np.array(ys)))

                    if self.centroid is None:
                        self.centroid_mov_vector = None
                        # create centroid
                        self.centroid = (new_centroid_x, new_centroid_y)
                    else:
                        self.centroid_mov_vector = (new_centroid_x-self.centroid[0], new_centroid_y-self.centroid[1])
                        self.reset_mask = False
                        # update centroid
                        self.centroid = (new_centroid_x, new_centroid_y)
                    cv.circle(vis, (self.centroid[0], self.centroid[1]), 5, (0, 255, 0), -1)
                self.tracks = new_tracks

                draw_str(vis, (20, 20), 'track count: %d, frame: %d' % (len(self.tracks), self.frame_idx))
            elif len(self.tracks) == 0:
                self.reset_mask = True

            # every 'self.detect_interval' frames compute ORB points
            if self.frame_idx % self.detect_interval == 0:
                # create mask
                mask = np.zeros_like(frame_gray)
                if self.reset_mask:
                    mask[:, :] = 255
                else:
                    mask = self.smart_mask(mask)
                cv.imshow('mask', mask)
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

    @staticmethod
    def reset_mask(mask):
        mask[:, :] = 255
        return mask

    def smart_mask(self, mask):
        mask_size = 100
        pred_centroid = (self.centroid[0] + self.centroid_mov_vector[0],
                         self.centroid[1] + self.centroid_mov_vector[1])
        mask[np.clip(pred_centroid[1] - mask_size//2, a_min=0, a_max=mask.shape[1]):np.clip(pred_centroid[1] + mask_size//2, a_min=0, a_max=mask.shape[1]),
             np.clip(pred_centroid[0] - mask_size//2, a_min=0, a_max=mask.shape[0]):np.clip(pred_centroid[0] + mask_size//2, a_min=0, a_max=mask.shape[0])] = 255
        return mask

    @staticmethod
    def draw_flow_field(fr1_features, fr2_features, frame):
        for (pX, pY), (qX, qY) in zip(fr1_features.reshape(-1, 2), fr2_features.reshape(-1, 2)):
            angle = np.arctan2(pY - qY, pX - qX)
            hypotenuse = np.sqrt(np.square(pY - qY) + np.square(pX - qX))

            qX = int(pX - 4 * hypotenuse * np.cos(angle))
            qY = int(pY - 4 * hypotenuse * np.sin(angle))

            cv.line(frame, (pX, pY), (qX, qY), (0, 0, 255), 1, cv.LINE_AA, 0)

            pX = int(qX + 9 * np.cos(angle + np.pi / 4))
            pY = int(qY + 9 * np.sin(angle + np.pi / 4))
            cv.line(frame, (pX, pY), (qX, qY), (0, 0, 255), 1, cv.LINE_AA, 0)
            pX = int(qX + 9 * np.cos(angle - np.pi / 4))
            pY = int(qY + 9 * np.sin(angle - np.pi / 4))
            cv.line(frame, (pX, pY), (qX, qY), (0, 0, 255), 1, cv.LINE_AA, 0)

            return frame

    @staticmethod
    def crop(frame):
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
        video_src = "../../data/videos_2D/case-1-2D.avi"
        '''
        Note:
        for clip-case-2 - NOT SO GOOD - Very, very, very difficult video
        for clip-case-3 - GOOD
        for clip-pat-1-video-4 - GOOD
        for clip-pat-1-video-6 - GOOD - Tool's body interference
        for clip-pat-2-video-2 - VERY GOOD
        for clip-pat-2-video-6 - NOT SO GOOD - Glare interference
        '''
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
