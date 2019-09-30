import os
from ast import literal_eval

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json
from scipy.spatial import distance
from sklearn.utils import shuffle
from tqdm import tqdm

base_path = '../../data/datasets/all_distance_frames/'


def load_images(filenames):
    # images container
    images = []

    for f in filenames:
        img = cv.imread(os.path.join(base_path, f))
        images.append(img)

    return np.array(images)


def load_targets(targets):
    out = []
    for x in targets:
        out.append(np.array(literal_eval(x)))
    return np.asarray(out, dtype=np.float64)


def main():
    df = pd.read_csv('../../data/targets/targets-v2.csv', sep=';')
    df = shuffle(df[df['p'] != '(-1, -1)'])
    sample = df.sample(n=300)

    x = load_images(sample['file'].tolist())
    y = load_targets(sample['p'].tolist())

    json_file = open('./trained_model/tool_09-30-17_02.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./trained_model/tool_09-30-17_02.h5")

    pred_y = loaded_model.predict(x / 255.0, batch_size=1)
    error = []

    for p in pred_y:
        p[0] = p[0] * 240
        p[1] = p[1] * 320

    i = 0
    for pred, im, target in tqdm(zip(pred_y, x, y)):
        error.append(distance.euclidean(pred, target))
        im = im.astype(np.float32)
        # draw prediction
        cv.circle(im, (int(pred[0]), int(pred[1])), 1, (0, 0, 255), -1, cv.LINE_AA)
        # draw ground truth
        cv.circle(im, (int(target[0]), int(target[1])), 1, (0, 255, 0), -1, cv.LINE_AA)
        cv.putText(im, 'pred:(' + str(pred[0] * 240) + ',' + str(pred[1] * 320) + ')', (20, 220),
                   cv.FONT_HERSHEY_PLAIN, .6,
                   color=(0, 0, 255))
        cv.putText(im, 'real:(' + str(target[0]) + ',' + str(target[1]) + ')', (20, 200),
                   cv.FONT_HERSHEY_PLAIN, .6,
                   color=(0, 255, 0))

        cv.imwrite('./preds/pred%d.png' % i, im)

        i += 1

    avg_error = float(np.mean(error, dtype=np.float64))
    std_error = float(np.std(error, dtype=np.float64))
    print(avg_error)
    print(std_error)

    labels = sample['file'].to_list()
    labels = [l[:-4] for l in labels]
    plt.figure()
    plt.title('Error distribution - Avg: %.2f - Std: %.2f' % (avg_error, std_error))
    plt.scatter(labels, error)
    plt.xticks(rotation=90, fontsize=5)
    plt.savefig('./training_outputs/error_distr')


if __name__ == '__main__':
    main()
