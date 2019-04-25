from keras.models import model_from_json
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def main():
    test_path = '../../../data/datasets/all_distance_frames/'
    test = pd.read_csv('../../../data/targets/test.csv', sep=';')
    test['valid'] = test['valid'].astype('str')
    test = test.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # load json and create model
    json_file = open('./model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./model.h5")
    print("Loaded model from disk")

    pred = test[:50]
    x_pred = np.zeros(shape=[50, 240, 320, 3])
    for i, r in tqdm(pred.iterrows()):
        r['file'] = '' + test_path + '/' + r['file']
        x_pred[i] = cv2.imread(r['file'])

    x_pred = x_pred.reshape([50, 320, 240, 3])
    print(x_pred.shape)
    y_pred = loaded_model.predict_classes(x_pred)

    pred['predicted'] = y_pred

    print(pred)

    for _, r in tqdm(pred.iterrows()):
        plt.figure()
        plt.imshow(cv2.cvtColor(cv2.imread(r['file']).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title('ground_truth=%s, predicted=%s' % (r['valid'], r['predicted']))
        plt.show()


if __name__ == '__main__':
    main()
