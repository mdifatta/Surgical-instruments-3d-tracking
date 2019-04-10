import cv2

def main():
    print('starting main ...')
    cap = cv2.VideoCapture('../../data/videos/case-1-3D.avi')
    i = 0
    while cap.isOpened():
        ret, f = cap.read()
        if ret:
            cv2.imwrite("../..data/datasets/full3d/frame-%d.png" % i, f)
            i = i + 1
        else:
            break

    cap.release()


if __name__ == "__main__":
    main()
