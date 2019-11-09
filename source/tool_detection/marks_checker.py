import cv2
import pandas as pd


def checker():
    name = 'case-2'

    path = '../../data/datasets/2d_frames_folders/%s/' % name
    df = pd.read_csv('../../data/targets/%s.csv' % name, sep=';')

    for _, r in df.iterrows():
        tmp = cv2.imread(path + r['file'])
        if isinstance(eval(r['p1']), tuple):
            tmp = cv2.circle(tmp, eval(r['p1']), 2, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            tmp = cv2.circle(tmp, eval(r['p2']), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.imshow(r['file'], tmp)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    checker()
