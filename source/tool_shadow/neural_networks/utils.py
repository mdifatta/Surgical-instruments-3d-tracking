from imgaug import augmenters as iaa
import cv2
import random


def augmentation(images):
    cv2.imshow('test', images)
    cv2.waitKey()
    rnd = random.uniform(0, 1)
    if rnd > .6:
        flipper = iaa.Fliplr(1)
        # TODO: change coordinates
        images = flipper.augment_image(images)
        cv2.imshow('test', images)
        cv2.waitKey()

    augmenter = iaa.Sometimes(0.6,
                              iaa.SomeOf(3,
                                         [iaa.Add((-40, 40), per_channel=0.4),
                                          iaa.Multiply((0.5, 1.5), per_channel=0.4),
                                          iaa.Dropout(p=(0, 0.01), per_channel=True),
                                          iaa.CoarseDropout(p=(0, 0.01), per_channel=True, size_percent=.5)],
                                         random_order=True
                                         )
                              )
    return augmenter.augment_image(images)


if __name__ == '__main__':
    img = cv2.imread('../../../data/datasets/2d_frames_folders/prova/frame-0.png')

    for i in range(10):
        ret = augmentation(img)

        cv2.imshow('test', ret)

        cv2.waitKey()
