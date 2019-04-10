import cv2
from utilities import normalize


def disparity_computation(left_img, right_img):

    stereo = cv2.StereoSGBM_create(minDisparity=-15,
                                   numDisparities=96,
                                   blockSize=1,
                                   speckleRange=2,
                                   speckleWindowSize=4
                                   )

    disparity = stereo.compute(left_img, right_img)

    return normalize(disparity)


def main():
    cap = cv2.VideoCapture("../../data/videos/case-1-3D.avi")

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
    if ret:
        # downscale for faster computation
        frame = cv2.pyrDown(frame)

        # compute disparity map
        mid = int(frame.shape[1] / 2)
        disparity = disparity_computation(
            frame[:, 0:mid, :],
            frame[:, mid:frame.shape[1], :]
        )

        # compute depth map from disparity map

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
