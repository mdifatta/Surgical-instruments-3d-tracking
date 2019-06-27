from keras.models import model_from_json
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm


def main():
    test_path = '../../../data/datasets/all_distance_frames/'
    test = pd.read_csv('../../../data/targets/test.csv', sep=';')
    test['valid'] = test['valid'].astype('str')
    test = test.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # load json and create model
    json_file = open('./mobilev2_b32_lr0001ds.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./mobilev2_b32_lr0001ds.h5")
    print("Loaded model from disk")

    pred = test
    x_pred = np.zeros(shape=[len(pred.index), 240, 320, 3])
    for i, r in pred.iterrows():
        r['file'] = '' + test_path + '/' + r['file']
        x_pred[i] = cv2.imread(r['file'])

    x_pred = x_pred.reshape([len(pred.index), 320, 240, 3])

    #y_pred = loaded_model.predict_classes(x_pred)
    y_pred = np.argmax(loaded_model.predict(x_pred), axis=1)

    pred['predicted'] = y_pred
    y_test = []
    for i, r in pred.iterrows():
        y_test.append(int(r['valid']))

    y_test = np.array(y_test)
    '''
    for _, r in tqdm(pred.iterrows()):
        plt.figure()
        plt.imshow(cv2.cvtColor(cv2.imread(r['file']), cv2.COLOR_BGR2RGB))
        plt.title('ground_truth=%s, predicted=%s' % (r['valid'], r['predicted']))
        plt.show()
    '''

    plot_confusion_matrix(y_test, y_pred)
    plt.show()

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=['0', '1'], yticklabels=['0', '1'],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    main()
