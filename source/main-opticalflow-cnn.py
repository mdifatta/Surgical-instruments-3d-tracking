from __future__ import print_function

import cv2 as cv
import numpy as np
from keras.models import model_from_json
from scipy import signal


class StereoParams:
    # ################## sets of params for good disparities ##################
    fast_stereo_match_params = dict(
        # resize > CLAHE > gray
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
        minDisparity=-7,
        numDisparities=96,
        blockSize=5,
        speckleRange=5,
        speckleWindowSize=180,
        disp12MaxDiff=5,
        P1=200,
        P2=1200,
        uniquenessRatio=6
    )

    params_no_sharp_bgr = dict(minDisparity=-6,
                               numDisparities=64,
                               blockSize=9,
                               speckleRange=7,
                               speckleWindowSize=180,
                               disp12MaxDiff=13,
                               mode=cv.STEREO_SGBM_MODE_HH
                               )

    params_no_sharp_bgr_P1_P2 = dict(minDisparity=-7,
                                     numDisparities=112,
                                     blockSize=7,  # or 9
                                     speckleRange=15,
                                     speckleWindowSize=200,
                                     disp12MaxDiff=40,
                                     # mode=cv.STEREO_SGBM_MODE_HH,
                                     P1=250,
                                     P2=5000  # or 7000
                                     )

    params_clahe_bgr_P1_P2 = dict(minDisparity=-7,
                                  numDisparities=112,
                                  blockSize=5,
                                  speckleRange=8,
                                  speckleWindowSize=200,
                                  disp12MaxDiff=5,
                                  P1=1,
                                  P2=4000,
                                  uniquenessRatio=4
                                  )

    params_clahe_bgr_P1_P2_v2 = dict(
                                     minDisparity=-4,
                                     numDisparities=128,
                                     blockSize=3,
                                     speckleRange=5,
                                     speckleWindowSize=150,
                                     disp12MaxDiff=7,
                                     P1=20,
                                     P2=5000,
                                     uniquenessRatio=7
                                    )

    params_no_sharp_gray = dict(minDisparity=5,
                                numDisparities=80,
                                blockSize=13,
                                speckleRange=7,
                                speckleWindowSize=130,
                                disp12MaxDiff=50,
                                mode=cv.STEREO_SGBM_MODE_HH
                                )

    params_sharp_bgr = dict(minDisparity=-16,  # try to increase over 0
                            numDisparities=80,
                            blockSize=7,
                            speckleRange=6,
                            speckleWindowSize=130,
                            disp12MaxDiff=25,  # try to increase over 30/40
                            mode=cv.STEREO_SGBM_MODE_HH
                            )

    params_sharp_bgr_P1_P2 = dict(minDisparity=-4,
                                  numDisparities=112,
                                  blockSize=5,
                                  speckleRange=10,
                                  speckleWindowSize=200,
                                  disp12MaxDiff=25,
                                  P1=250,
                                  P2=2000
                                  )


class App:
    # Lucas-Kanade algorithm's params
    lk_params = dict(winSize=(15, 15),
                     maxLevel=7,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    SAFETY_THRESHOLD = 13
    DISPARITY_MAP_ERROR_CODE = -1
    UNSAFE_ERROR_CODE = 0
    OK_CODE = 1

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
        # select starting frame
        self.cam.set(cv.CAP_PROP_POS_FRAMES, 0)
        # starting frame index
        self.frame_idx = 0
        # ORB key-points detector
        self.orb = cv.ORB_create(nfeatures=10)
        # centroid
        self.centroid = ()
        # frames count
        self.frames_count = self.cam.get(cv.CAP_PROP_FRAME_COUNT)
        # previous gray frame for quality control
        self.prev_gray = None
        # initialize CLAHE object
        self.claher = cv.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        # initialize stereo matcher object
        self.stereo_matcher = cv.StereoSGBM_create(
            **StereoParams.fast_stereo_match_params
        )
        self.curr_frame = None
        model_file = './tool_shadow/trained_model/tool_10-16-10-15.json'
        weights_file = './tool_shadow/trained_model/weights_checkpoint_10-16-10-15.h5'
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(weights_file)
        print("Loaded model from disk")

    def run(self):
        left_edge, right_edge = 0, -1

        if not self.cam.isOpened():
            print('Error reading video')
            return
        while self.cam.isOpened():
            # read next frame
            _ret, full_frame = self.cam.read()
            if not _ret:
                self.frame_idx += 1
                continue
            # crop frame to remove TrueVision logo which interferes with ORB detection and stereo matching
            full_frame = full_frame[:1026, :, :]
            # divide left and right frame from the stereo frame
            mid = int(full_frame.shape[1] / 2)
            left_frame = full_frame[:, 0:mid, :]
            right_frame = full_frame[:, mid:, :]
            # get a copy of the left frame as reference for the Optical Flow
            frame = left_frame
            if self.frame_idx == 0:
                left_edge, right_edge = self.crop(frame, width=1368)
            # crop black bands from frames' edges
            left_frame = left_frame[:, left_edge:right_edge, :]
            right_frame = right_frame[:, left_edge:right_edge, :]
            self.curr_frame = left_frame
            frame_gray = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)
            # create a working copy of the current frame
            vis = left_frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                # reshape self.tracks
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # compute the optical flow for the current frame
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                # compute the counter-proof optical flow for the previous frame
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
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

                if self.centroid:
                    outcome, distance = self.depth_estimation(left_frame, right_frame, self.centroid)
                    if outcome == App.OK_CODE:
                        self.draw_str(vis, (20, 20), 'track count: %d, frame: %d, dist: %.2f' %
                                      (len(self.tracks), self.frame_idx, distance))
                    elif outcome == App.DISPARITY_MAP_ERROR_CODE:
                        self.draw_str(vis, (20, 20), 'track count: %d, frame: %d, DISP MAP QUALITY INSUFFICIENT' %
                                      (len(self.tracks), self.frame_idx))
                    else:
                        self.draw_str(vis, (20, 20), 'track count: %d, frame: %d, DISTANCE WARNING' %
                                      (len(self.tracks), self.frame_idx))
                else:
                    self.draw_str(vis, (20, 20), 'track count: %d, frame: %d' % (len(self.tracks), self.frame_idx))

            # every 'self.detect_interval' frames compute ORB points
            if self.frame_idx % self.detect_interval == 0:
                # create mask
                # TODO: try to provide a smarter mask
                # mask = np.zeros(shape=(frame.shape[0], frame.shape[1]))
                mask = np.zeros_like(frame_gray)
                # mask = self.smart_mask(mask, frame_gray)
                mask[:, :] = 255
                # detect ORB points
                self.detect_tip()
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

            ch = cv.waitKey(1000 // self.fps)
            if ch == 27:
                break

        self.cam.release()
        cv.destroyAllWindows()

    def smart_mask(self, mask, frame):
        smart_mask = np.copy(mask)
        if self.frame_idx == 0:
            roi = cv.selectROI('Choose ROI', frame, fromCenter=False)
            smart_mask[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = 255
        else:
            smart_mask[:] = 255
        return smart_mask

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
    def crop(frame, width=1368):
        _, thr = cv.threshold(cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
                              0,
                              255,
                              cv.THRESH_BINARY + cv.THRESH_OTSU
                              )
        crop_mask = cv.findNonZero(thr)

        left_edge = crop_mask[:, 0, 0].min()
        right_edge = crop_mask[:, 0, 0].max()

        roi = right_edge - left_edge
        delta2target = width - roi
        padding = delta2target // 2

        of_left = (left_edge - padding < 0)
        of_right = (right_edge + padding > frame.shape[1])

        if of_left and of_right:
            return 'ERROR'

        if of_left and not of_right:
            left_margin = 0
            remainder = (padding - left_edge)
            right_margin = right_edge + padding + remainder
        elif of_right and not of_left:
            right_margin = frame.shape[1]
            remainder = (right_edge + padding - frame.shape[1])
            left_margin = left_edge - padding - remainder
        else:
            right_margin = right_edge + padding
            left_margin = left_edge - padding

        return left_margin, right_margin

    @staticmethod
    def draw_str(dst, target, s):
        x, y = target
        cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
        cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

    def depth_estimation(self, left, right, centroid):

        # match left and right frames
        disparity = self.stereo_matcher.compute(
            cv.resize(self.claher.apply(cv.cvtColor(left, cv.COLOR_BGR2GRAY)), (0, 0), fx=.6, fy=.6),
            cv.resize(self.claher.apply(cv.cvtColor(right, cv.COLOR_BGR2GRAY)), (0, 0), fx=.6, fy=.6)
        )

        # values from matching are float, normalize them between 0-255 as integer
        disparity = cv.normalize(cv.resize(disparity, (0, 0), fx=1/.6, fy=1/.6), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        code, delta_disparity = App.compare_disparities(disparity, centroid)

        return code, delta_disparity

    def histogram_equalizer(self, img):
        l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))
        l_channel = self.claher.apply(l)
        lab = cv.merge((l_channel, a, b))
        return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    @staticmethod
    def compare_disparities(disparity_map, centroid, tip_mask_size=100, retina_mask_size=320):
        # TODO: dynamic size of the masks

        # filter masks from the disparity map
        tip = disparity_map[centroid[1] - (tip_mask_size // 2):centroid[1] + (tip_mask_size // 2),
                            centroid[0] - (tip_mask_size // 2):centroid[0] + (tip_mask_size // 2)]
        retina = disparity_map[centroid[1] - (retina_mask_size // 2):centroid[1] + (retina_mask_size // 2),
                               centroid[0] - (retina_mask_size // 2):centroid[0] + (retina_mask_size // 2)]

        if retina.shape != (320, 320):
            return App.DISPARITY_MAP_ERROR_CODE, None
        else:
            # compute average disparity for the tool's tip
            tip_avg_disparity = App.tip_averaging(tip)
            # compute average disparity for the background retina
            retina_avg_disparity = App.retina_averaging(retina, tip_avg_disparity)

            # estimated delta disparity between the tool's tip and the retina in background
            # CAVEAT: delta_disparity should always be > 0;
            # if it's =0 we're operating dangerously, we should avoid to get to 0
            # if it's <0 we have some serious problems
            delta_disparity = tip_avg_disparity - retina_avg_disparity
            if delta_disparity < 0:
                return App.DISPARITY_MAP_ERROR_CODE, delta_disparity
            elif delta_disparity <= App.SAFETY_THRESHOLD:
                return App.UNSAFE_ERROR_CODE, delta_disparity
            else:
                return App.OK_CODE, delta_disparity

    @staticmethod
    def gaussian_weights(kernel_len, std=2.5):
        """Returns a 2D Gaussian kernel array."""
        gauss_kernel_1d = signal.gaussian(kernel_len, std=std).reshape(kernel_len, 1)
        gauss_kernel_2d = np.outer(gauss_kernel_1d, gauss_kernel_1d)
        return gauss_kernel_2d

    @staticmethod
    def tip_averaging(tip):
        # Gaussian weights for the tip
        weights = App.gaussian_weights(kernel_len=100)

        weights[tip <= 20] = 1e-7
        # weighted average disparity value for the tip
        avg_tip = np.average(tip, weights=weights)
        return avg_tip

    @staticmethod
    def retina_averaging(retina, tip_avg):
        # clip retina's disparities
        retina = np.clip(retina, 1e-7, int(tip_avg))

        # weights for the background
        weights = np.full_like(retina, fill_value=tip_avg, dtype=np.uint8)
        weights = np.clip(np.subtract(weights, retina), a_min=1e-7, a_max=int(tip_avg))
        # get rid of meaningless areas
        weights[retina < 10] = 1e-7
        # get rid of white noise near edges and outliers
        weights[retina >= tip_avg] = 1e-7
        # normalize weights
        weights = cv.normalize(weights, None, 1e-7, 1, cv.NORM_MINMAX, cv.CV_32F)
        # weighted average and median disparity value for the retina
        avg_retina = np.average(retina, weights=weights)
        # add a penalty, i.e. increase it by 7%, the retina avg value to
        # take into account noise and to have a larger margin of error
        avg_retina_w_penalty = avg_retina * 1.07
        return avg_retina

    def detect_tip(self):
        # resize image
        curr_img = cv.resize(self.curr_frame, (320, 240))
        # expand dims to include batch size
        _input = np.expand_dims(curr_img, axis=0)
        # run CNN
        pred = self.loaded_model.predict(_input / 255.0, batch_size=1)
        # re-scale predictions
        pred[0][0] = pred[0][0] * 240
        pred[0][1] = pred[0][1] * 320
        # show predicted point for test
        cv.imshow('test', cv.circle(curr_img, (int(pred[0][0]), int(pred[0][1])), 2, (255, 0, 0), -1, cv.LINE_AA))
        cv.waitKey()
        cv.destroyWindow('test')
        print(pred * 4.275)


def main():
    try:
        # 2D video
        # video_src = "../data/videos/case-12.mp4"
        # Note:
        # for clip-case-2 - TRACKING NOT SO GOOD (Very, very, very difficult video)
        # for clip-case-3 - TRACKING GOOD
        # for clip-pat-1-video-4 - TRACKING GOOD
        # for clip-pat-1-video-6 - TRACKING GOOD (Tool's body interference)
        # for clip-pat-2-video-2 - TRACKING VERY GOOD
        # for clip-pat-2-video-6 - TRACKING NOT SO GOOD (Glare interference)
        # 3D video
        video_src = "../data/videos/case-1-3D.avi"
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
