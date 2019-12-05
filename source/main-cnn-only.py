from __future__ import print_function

import PySimpleGUI as Sg
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


class GUI:
    def __init__(self, num_frames):

        layout = [
            [Sg.Image(filename='', key='_IMAGE_', size=(866, 650))],
            [Sg.Slider(range=(1, num_frames), orientation='h', size=(85, 10), default_value=0, enable_events=True,
                       key='_SLIDER_'), Sg.Button('Pause', focus=True), Sg.Button('Quit')]
        ]

        self.window = Sg.Window('Real-time', location=(0, 0)).Layout(layout).Finalize()

    def update_frame(self, img):
        encoded_img = cv.imencode('.png', cv.resize(img, (866, 650)))[1].tobytes()
        self.window['_IMAGE_'].update(data=encoded_img)

    def update_slider(self, pos):
        self.window['_SLIDER_'].update(pos)


class App:
    # safety threshold, this value should be tuned
    SAFETY_THRESHOLD = 21

    # outcome codes
    DISPARITY_MAP_ERROR_CODE = -1
    UNSAFE_ERROR_CODE = 0
    OK_CODE = 1

    RED = (0, 0, 255)
    ORANGE = (0, 128, 255)
    YELLOW = (0, 255, 255)
    L_GREEN = (0, 255, 128)
    GREEN = (0, 255, 0)

    # max length of retina history
    # CAVEAT: the videos' frame rate is 60 fps so with =60 a memory of 1 sec in kept,
    # with =120 a memory of 2 sec is kept and so on
    MAX_HISTORY_COUNT = 120

    # max variance for a retina disparity value
    RETINA_VAR_THRESHOLD = 4

    def __init__(self, video_src):
        # input video's fps
        self.fps = 24
        # OpenCV's video reader
        self.cam = cv.VideoCapture(video_src)
        self.cam.set(cv.CAP_PROP_POS_FRAMES, 0)
        # starting frame index
        self.frame_idx = 0
        # centroid
        self.tooltip = np.array([])
        # set video frame rate
        self.cam.set(cv.CAP_PROP_FPS, self.fps)
        # frames count
        self.frames_count = self.cam.get(cv.CAP_PROP_FRAME_COUNT)
        # initialize CLAHE object
        self.claher = cv.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
        # initialize stereo matcher object
        self.stereo_matcher = cv.StereoSGBM_create(
            **StereoParams.fast_stereo_match_params
        )
        # current frame
        self.curr_frame = None

        # store last disparity values for the retina to detect outliers
        self.retina_history = []
        # partial sum of last disparity values for the retina
        self.retina_history_tot = 0
        # length of the retina's history
        self.retina_history_count = 0

        model_file = './tool_detection/trained_model/tool_10-16-10-15.json'
        weights_file = './tool_detection/trained_model/weights_checkpoint_10-16-10-15.h5'
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(weights_file)
        print("Loaded model from disk")

        self.GUI = GUI(num_frames=self.frames_count)

    def run(self):
        left_edge, right_edge = 0, -1

        if not self.cam.isOpened():
            print('Error reading video')
            return

        _PLAY = True

        while self.cam.isOpened():
            # read next frame
            _ret, full_frame = self.cam.read()
            if not _ret:
                self.frame_idx += 1
                continue
            # crop frame to remove TrueVision logo
            full_frame = full_frame[:1026, :, :]
            # divide left and right frame from the stereo frame
            mid = int(full_frame.shape[1] / 2)
            left_frame = full_frame[:, 0:mid, :]
            right_frame = full_frame[:, mid:, :]
            # get a copy of the left frame
            frame = left_frame
            if self.frame_idx == 0:
                left_edge, right_edge = self.crop(frame, width=1368)
            # crop black bands from frames' edges
            left_frame = left_frame[:, left_edge:right_edge, :]
            right_frame = right_frame[:, left_edge:right_edge, :]
            self.curr_frame = left_frame
            # create a working copy of the current frame
            vis = left_frame.copy()

            if self.tooltip:
                outcome, distance = self.depth_estimation(left_frame, right_frame, self.tooltip)
                if distance is None:
                    rect_color = App.ORANGE
                    rect_size = 15
                elif distance <= App.SAFETY_THRESHOLD:
                    rect_color = App.RED
                    rect_size = 25
                elif distance <= App.SAFETY_THRESHOLD + 7:
                    rect_color = App.ORANGE
                    rect_size = 50
                elif distance <= App.SAFETY_THRESHOLD + 14:
                    rect_color = App.YELLOW
                    rect_size = 100
                elif distance <= App.SAFETY_THRESHOLD + 21:
                    rect_color = App.L_GREEN
                    rect_size = 200
                else:
                    rect_color = App.GREEN
                    rect_size = 300

                cv.rectangle(vis, (15, 15), (rect_size, 100), rect_color, -1)
                cv.circle(vis, (self.tooltip[0], self.tooltip[1]), rect_size, rect_color, thickness=2)
                if outcome == App.DISPARITY_MAP_ERROR_CODE:
                    if distance is None:
                        self.draw_str(vis, (20, 20), 'DISP MAP QUALITY ISSUE')
                    else:
                        self.draw_str(vis, (20, 20), 'WARNING')
                elif outcome == App.UNSAFE_ERROR_CODE:
                    self.draw_str(vis, (480, 50), 'DISTANCE WARNING', font_size=3.0, color=(0, 0, 255))

            self.detect_tip()

            # read gui events
            gui_event, gui_values = self.GUI.window.read(timeout=1000 // self.fps)

            if gui_event in (None, 'Quit'):
                self.GUI.window.close()
                break

            if gui_event == 'Pause':
                _PLAY = not _PLAY

            if _PLAY:
                if int(gui_values['_SLIDER_']) != self.frame_idx - 1:
                    self.frame_idx = int(gui_values['_SLIDER_'])
                    self.cam.set(cv.CAP_PROP_POS_FRAMES, self.frame_idx)

                # increment frame index
                self.frame_idx += 1
                self.GUI.update_frame(vis)
                # update slider with new position
                self.GUI.update_slider(self.frame_idx)
            else:
                if int(gui_values['_SLIDER_']) != self.frame_idx - 1:
                    self.frame_idx = int(gui_values['_SLIDER_'])
                    self.cam.set(cv.CAP_PROP_POS_FRAMES, self.frame_idx)
                self.GUI.update_frame(vis)
                # update slider with new position
                self.GUI.update_slider(self.frame_idx)

        self.cam.release()

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
    def draw_str(dst, target, s, font_size=1.0, color=(255, 255, 255)):
        x, y = target
        cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), thickness=3, lineType=cv.LINE_AA)
        cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, font_size, color, thickness=2,  lineType=cv.LINE_AA)

    def depth_estimation(self, left, right, centroid):
        # match left and right frames
        disparity = self.stereo_matcher.compute(
            cv.resize(self.claher.apply(cv.cvtColor(left, cv.COLOR_BGR2GRAY)), (0, 0), fx=.6, fy=.6),
            cv.resize(self.claher.apply(cv.cvtColor(right, cv.COLOR_BGR2GRAY)), (0, 0), fx=.6, fy=.6)
        )
        # values from matching are float, normalize them between 0-255 as integer
        disparity = cv.normalize(cv.resize(disparity, (0, 0), fx=1/.6, fy=1/.6), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        code, delta_disparity = self.compare_disparities(disparity, centroid)

        return code, delta_disparity

    def histogram_equalizer(self, img):
        l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))
        l_channel = self.claher.apply(l)
        lab = cv.merge((l_channel, a, b))
        return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    def compare_disparities(self, disparity_map, centroid, tip_mask_size=30, retina_mask_size=320):

        # filter masks from the disparity map
        tip = disparity_map[centroid[1] - (tip_mask_size // 2):centroid[1] + (tip_mask_size // 2),
                            centroid[0] - (tip_mask_size // 2):centroid[0] + (tip_mask_size // 2)]
        retina = disparity_map[centroid[1] - (retina_mask_size // 2):centroid[1] + (retina_mask_size // 2),
                               centroid[0] - (retina_mask_size // 2):centroid[0] + (retina_mask_size // 2)]

        if retina.shape != (320, 320):
            return App.DISPARITY_MAP_ERROR_CODE, None
        else:
            # compute average disparity for the tool's tip
            tip_avg_disparity = App.tip_averaging(tip, tip_mask_size)
            # compute average disparity for the background retina
            retina_avg_disparity = App.retina_averaging(retina, tip_avg_disparity)

            _, retina_value = self.check_retina_history(retina_avg_disparity)

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
    def tip_averaging(tip, tip_mask_size):
        # Gaussian weights for the tip
        weights = App.gaussian_weights(kernel_len=tip_mask_size)

        # cv.imshow('weights', cv.normalize(weights, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))

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
        pred = pred.reshape(2)
        self.tooltip = tuple(np.rint(pred * 4.275).astype(int))

    def check_retina_history(self, value):
        # collect at least MAX_HISTORY_COUNT values before considering the average value
        if self.retina_history_count < App.MAX_HISTORY_COUNT:
            self.retina_history.append(value)
            self.retina_history_tot += value
            self.retina_history_count += 1
            return 0, value
        else:
            # compute the current average
            curr_avg = (self.retina_history_tot / self.retina_history_count)
            # if the current value is too far away (i.e. more than RETINA_VAR_THRESHOLD) from the average
            if np.abs(curr_avg - value) >= App.RETINA_VAR_THRESHOLD:
                # the current point is an outlier, return the average of the history
                return -1, curr_avg
            else:
                # remove oldest retina value from the history and from the sum
                oldest_value = self.retina_history.pop(0)
                self.retina_history_tot -= oldest_value
                # add the new value to the history and to the sum
                self.retina_history.append(value)
                self.retina_history_tot += value
                # return the updated average
                return 1, (self.retina_history_tot / self.retina_history_count)


def main():
    Sg.ChangeLookAndFeel('Black')
    video_src = Sg.PopupGetFile('Please enter a file name')

    while not (video_src is None):
        App(video_src).run()
        video_src = Sg.PopupGetFile('Please enter a file name')

    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
