from imgaug import augmenters as iaa
import imgaug as ia
import pandas as pd
from tqdm import tqdm
import cv2
import os


class Point:
    def __init__(self, point):
        self.x = int(point[0])
        self.y = int(point[1])

    def getx(self):
        return self.x

    def gety(self):
        return self.y


def resize_invalid(image):
    augmenter = iaa.Resize({
        'height': 240,
        'width': 320
    })

    return augmenter.augment_image(image)


def resize_valid(image, points):
    augmenter = iaa.Resize({
        'height': 240,
        'width': 320
    })

    key_points = ia.KeypointsOnImage([
        ia.Keypoint(x=points[0].getx(), y=points[0].gety()),
        ia.Keypoint(x=points[1].getx(), y=points[1].gety())
    ], shape=image.shape)

    augmenter = augmenter.to_deterministic()

    return augmenter.augment_image(image), augmenter.augment_keypoints(key_points)


if __name__ == '__main__':
    # ############### params ####################
    video_name = 'patient2-6' # MISSING, NOT RESIZED
    marked = False
    # ###########################################

    if marked:
        print('opening marked video ...')
        df = pd.read_csv('../../data/targets/' + video_name + '.csv', sep=';')
        file = open('../../data/targets/' + video_name + '.csv', 'w')
        file.write('file;valid;p1;p2;dist\n')
        file.close()
        for _, r in tqdm(df.iterrows()):
            img = cv2.imread('../../data/datasets/2d_frames_folders/' + video_name + '/' + r['file'])

            if r['valid'] == 1:
                # convert string tuple to instance of Point
                p1 = r['p1'][:len(r['p1']) - 1]
                p1 = p1.strip()[1:]
                p1 = p1.split(',')
                p1 = Point(p1)
                p2 = r['p2'][:len(r['p2']) - 1]
                p2 = p2.strip()[1:]
                p2 = p2.split(',')
                p2 = Point(p2)
                # augment valid frame
                aug_img, aug_points = resize_valid(img,
                                                    [p1,
                                                     p2
                                                     ])

                file = open('../../data/targets/' + video_name + '.csv', 'a+')
                file.write('%s;1;(%d, %d); (%d, %d); %.1f\n'
                           % (r['file'], aug_points.keypoints[0].x, aug_points.keypoints[0].y, aug_points.keypoints[1].x,
                              aug_points.keypoints[1].y, r['dist']))
                cv2.imwrite('../../data/datasets/2d_frames_folders/' + video_name + '/' + r['file'],
                            aug_img)
                file.close()
            else:
                aug_img = resize_invalid(img)

                file = open('../../data/targets/' + video_name + '.csv', 'a+')
                file.write('%s;0;-1;-1;-1\n' % r['file'])
                cv2.imwrite('../../data/datasets/2d_frames_folders/' + video_name + '/' + r['file'],
                            aug_img)
                file.close()

    else:
        print('opening not marked video ...')
        frames_list = os.listdir('../../data/datasets/2d_frames_folders/' + video_name + '/')
        for f in tqdm(frames_list):
            img = cv2.imread('../../data/datasets/2d_frames_folders/' + video_name + '/' + f)
            aug_img = resize_invalid(img)
            cv2.imwrite('../../data/datasets/2d_frames_folders/' + video_name + '/' + f, aug_img)