from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import pandas as pd
from tqdm import tqdm


class Point:
    def __init__(self, point):
        self.x = int(point[0])
        self.y = int(point[1])

    def getx(self):
        return self.x

    def gety(self):
        return self.y


def augment_invalid(image):
    augmenter = iaa.Sequential([
        # geometry
        iaa.Fliplr(0.4),
        iaa.Flipud(0.4),
        # color
        iaa.Add((-30, 30), per_channel=0.4),
        iaa.Multiply((0.5, 1.5), per_channel=0.4),
        iaa.Dropout(p=(0, 0.004), per_channel=True)
    ],
        random_order=True
    )

    augmenter = augmenter.to_deterministic()

    return augmenter.augment_image(image)


def augment_valid(image, points):

    key_points = ia.KeypointsOnImage([
        ia.Keypoint(x=points[0].getx(), y=points[0].gety()),
        ia.Keypoint(x=points[1].getx(), y=points[1].gety())
    ], shape=image.shape)

    augmenter = iaa.Sequential([
                      # geometry
                      iaa.Fliplr(0.4),
                      iaa.Flipud(0.4),
                      # color
                      iaa.Add((-30, 30), per_channel=0.6),
                      iaa.Multiply((0.5, 1.2), per_channel=0.3),
                      iaa.Dropout(p=(0, 0.004), per_channel=True)
    ],
        random_order=True
    )

    augmenter = augmenter.to_deterministic()

    return augmenter.augment_image(image), augmenter.augment_keypoints(key_points)


if __name__ == '__main__':
    # ################## PARAMS ######################
    original_frames = 'patient1-6'
    id = 7203
    # ###############################################

    file = open('../../../data/targets/augmented-' + original_frames + '.csv', 'a+')
    file.write('file;valid;p1;p2;dist\n')
    file.close()
    df = pd.read_csv('../../../data/targets/' + original_frames + '.csv', sep=';')

    for _, r in tqdm(df.iterrows()):
        img = cv2.imread('../../../data/datasets/2d_frames_folders/' + original_frames + '/' + r['file'])
        if r['valid'] == 1:
            # convert string tuple to instance of Point
            p1 = r['p1'][:len(r['p1'])-1]
            p1 = p1[1:]
            p1 = p1.split(',')
            p1 = Point(p1)
            p2 = r['p2'][:len(r['p2']) - 1]
            p2 = p2[1:]
            p2 = p2.split(',')
            p2 = Point(p2)
            # augment valid frame
            aug_img, aug_points = augment_valid(img,
                                                [p1,
                                                 p2
                                                 ])

            # save augmented image with new target
            file = open('../../../data/targets/augmented-' + original_frames + '.csv', 'a+')
            cv2.imwrite('../../../data/datasets/2d_frames_folders/augmented-%s/frame%d.png' % (original_frames, id),
                        aug_img)
            file.write('frame%d.png;1;(%d, %d); (%d, %d); %.1f\n'
                       % (id, aug_points.keypoints[0].x, aug_points.keypoints[0].y, aug_points.keypoints[1].x,
                          aug_points.keypoints[1].y, r['dist']))
            file.close()

            # draw old points on original image
            # tmp = cv2.circle(img, (p1.getx(), p1.gety()), 4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            # tmp = cv2.circle(tmp, (p2.getx(), p2.gety()), 4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            # cv2.imshow('original', tmp)
            # cv2.waitKey()
            # draw new points on augmented image
            # tmp = cv2.circle(aug_img, (aug_points.keypoints[0].x, aug_points.keypoints[0].y), 4, (255, 0, 0), -1,
            #                  lineType=cv2.LINE_AA)
            # tmp = cv2.circle(tmp, (aug_points.keypoints[1].x, aug_points.keypoints[1].y), 4, (255, 0, 0), -1,
            #                  lineType=cv2.LINE_AA)
            # cv2.imshow('points', tmp)
            # cv2.waitKey()
        else:
            # show original and augmented image, both without points
            aug_img = augment_invalid(img)

            # save augmented image
            file = open('../../../data/targets/augmented-' + original_frames + '.csv', 'a+')
            cv2.imwrite('../../../data/datasets/2d_frames_folders/augmented-%s/frame%d.png' % (original_frames, id),
                        aug_img)
            file.write('frame%d.png;0;-1;-1;-1\n' % id)
            file.close()
            # cv2.imshow('original', img)
            # cv2.waitKey()
            # cv2.imshow('points', aug_img)
            # cv2.waitKey()

        id = id + 1
