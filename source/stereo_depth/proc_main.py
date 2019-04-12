import cv2
from utilities import normalize
import matplotlib.pyplot as plt


def disparity_computation(left_img, right_img):

    stereo = cv2.StereoSGBM_create(minDisparity=-10,
                                   numDisparities=80,
                                   blockSize=1,
                                   speckleRange=2,
                                   speckleWindowSize=4
                                   )

    disparity = stereo.compute(left_img, right_img)

    return normalize(disparity)


def crop(left, right, width=1440):
    ret2, thr_left = cv2.threshold(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                                   0,
                                   255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                   )

    ret3, thr_right = cv2.threshold(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                    )

    mask_left = cv2.findNonZero(thr_left)

    left_edge = mask_left[:, 0, 0].min()
    right_edge = mask_left[:, 0, 0].max()

    delta = right_edge - left_edge
    delta2 = width - delta
    left_margin = int(delta2 / 2)
    right_margin = delta2 - left_margin

    left_cropped = left[:, left_edge - left_margin:right_edge + right_margin, :]

    mask_right = cv2.findNonZero(thr_right)

    left_edge = mask_right[:, 0, 0].min()
    right_edge = mask_right[:, 0, 0].max()

    delta = right_edge - left_edge
    delta2 = width - delta
    left_margin = int(delta2 / 2)
    right_margin = delta2 - left_margin

    right_cropped = right[:, left_edge - left_margin:right_edge + right_margin, :]

    return left_cropped, right_cropped


def main():
    cap = cv2.VideoCapture("../../data/videos/case-1-3D.avi")

    ret, frame = cap.read()
    if ret:
        # computer image's half
        mid = int(frame.shape[1] / 2)

        # divide left and right frames and crop TrueVision logo
        left_frame = frame[:, 0:1750, :]
        right_frame = frame[:, mid:mid + 1750, :]

        # crop image to shape [1080, width]
        left_cropped, right_cropped = crop(left_frame, right_frame, width=1440)

        # downscale for faster computation
        new_left = cv2.pyrDown(left_cropped)
        new_right = cv2.pyrDown(right_cropped)

        # compute disparity map
        disparity = disparity_computation(
            new_left,
            new_right
        )

        #plt.figure()
        #plt.imshow(disparity, 'gray')
        #splt.show()

    else:
        print('Error.')

    cap.release()


def colab_main():
    # ######################################################
    # ########  function to be run on Google Colab #########
    # ######################################################
    cap = cv2.VideoCapture("../case-1-3D.avi")

    '''
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.pyrDown(frame)
            mid = int(frame.shape[1] / 2)
            disparity = disparity_computation(
                                            frame[:, 0:mid, :],
                                            frame[:, mid:frame.shape[1], :]
                                              )
            # compute depth map from disparity map
            #writer.write(disparity.astype('uint8'))
            cv2.imshow('frame', disparity)
            cv2.waitKey(1)
        else:
            break
    '''
    ret, frame = cap.read()
    mid = int(frame.shape[1] / 2)
    left_f = cv2.UMat(frame[:, 0:mid, :])
    right_f = cv2.UMat(frame[:, mid:frame.shape[1], :])
    if ret:
        # downscale for faster computation
        left_f = cv2.pyrDown(left_f)
        right_f = cv2.pyrDown(right_f)

        # compute disparity map
        disparity = disparity_computation(
            left_f,
            right_f
        )

        # compute depth map from disparity map

    else:
        print('Error.')

    cap.release()


if __name__ == "__main__":
    main()
