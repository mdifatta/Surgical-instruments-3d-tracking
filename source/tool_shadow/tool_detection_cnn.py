import argparse
import datetime
import math
import os
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from keras import backend as K
from keras.activations import relu
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import sgd
from keras.regularizers import l2
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class MyGenerator(Sequence):
    def __init__(self, image_filenames, targets: pd.DataFrame, batch_size, resize_target, base_path: str):
        self.image_filenames, self.targets = image_filenames, self.process_targets(targets)
        self.batch_size = batch_size
        self.resize_target = resize_target
        self.basePath = base_path

    def __len__(self):
        return math.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        for filename in batch_x:
            img = cv.imread(os.path.join(self.basePath, filename))
            img = self.process_data(img)

            images.append(img)

        return np.array(images)/255.0, np.array(batch_y)

    def process_data(self, img: np.array):
        img = cv.resize(img, self.resize_target)

        return img

    def process_targets(self, df: pd.DataFrame):
        targets = []
        df['p'] = [eval(x) for x in df['p']]
        for x in df['p']:
            targets.append(np.array(x))

        targets = np.asarray(targets, dtype=np.float64)

        for i in range(len(targets)):
            if np.all(targets[i] > 0.0):
                targets[i][0] = targets[i][0] / 240
                targets[i][1] = targets[i][1] / 320
        return targets


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation=relu, padding='same',
                     data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.05))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation=relu, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.1))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation=relu, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=250, activation=relu))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=250, activation=relu))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=2, activation=sigmoid))
    # model.add(LeakyReLU(alpha=.2))

    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def R2(y_true, y_pred):
    SS_res=K.sum(K.square(y_true-y_pred))
    SS_tot=K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def process_image(img: np.array, shape):
    img = cv.resize(img, shape)
    img = np.expand_dims(img, axis=2)

    return img


def load_images(df: pd.DataFrame, base_path, shape):
    print('Loading images ...')
    # images container
    images = []

    for index, row in tqdm.tqdm(df.iterrows()):
        filename = row['file']
        img = cv.imread(os.path.join(base_path, filename))
        img = process_image(img, shape)

        images.append(img)

    print('Images loaded.')

    return np.array(images)


def load_targets(df: pd.DataFrame):
    print('Loading targets ...')
    targets = []
    # df.loc[df['p2'] == '-1', 'p2'] = '(-1, -1)'
    # df['p'] = [eval(x) for x in df['p']]
    # df = df.drop(labels=['p1', 'p2', 'valid', 'dist'], axis=1)
    for x in df['p']:
        targets.append(np.array(x))
    print('Targets loaded.')
    return np.asarray(targets, dtype=np.float64)


def check_matches(images, targets):
    for i in range(images.shape[0]):
        t = np.copy(images[i, :, :, :])
        print(targets[i, 0])
        print(targets[i, 1])
        cv.circle(t, (int(targets[i, 0] * 240), int(targets[i, 1] * 320)), 3, (0, 0, 255), -1, cv.LINE_AA)
        cv.imshow('test', t)
        cv.waitKey()


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env", type=str, required=True,
                    help="system where to execute")
    args = vars(ap.parse_args())

    # ######################### HYPER-PARAMS ##############################
    batch_size = 64
    target_shape = (320, 240)
    learning_rate = .008
    momentum = .9
    input_shape = (240, 320, 3)
    ENV = args['env']
    # ###############################################################

    if ENV == 'local':
        # ######################## local ############################
        # create base path for images
        base_path = '../../data/datasets/all_distance_frames/'
        # read targets csv
        df = pd.read_csv('../../data/targets/targets-v2.csv', sep=';')
        # take only valid frames
        df = df[df['p'] != '(-1, -1)']
        # shuffle data
        df = shuffle(df)
        # get a subset of the lines for local testing
        df = df.iloc[:500, :]
        # local batch size to handle restricted dataset dimension
        batch_size = 10
    else:
        # ################# for colab ####################
        # create base path for images
        base_path = './frames/'
        # read targets csv
        df = pd.read_csv('./targets/targets.csv', sep=';')
        # take only valid frames
        df = df[df['p'] != '(-1, -1)']
        # shuffle data
        df = shuffle(df)

    # split train and test as 90/10
    (train_df, test_df) = train_test_split(df,
                                           test_size=.1,
                                           train_size=.9)

    # split train and valid as 90/10 of previous test
    (train_df, valid_df) = train_test_split(train_df,
                                            test_size=.1,
                                            train_size=.9)

    # create generators for train ,test and valid
    train_generator = MyGenerator(
        image_filenames=train_df['file'].tolist(),
        targets=train_df,
        resize_target=target_shape,
        batch_size=batch_size,
        base_path=base_path
    )
    num_train_samples = len(train_df.index)

    test_generator = MyGenerator(
        image_filenames=test_df['file'].tolist(),
        targets=test_df,
        resize_target=target_shape,
        batch_size=batch_size,
        base_path=base_path
    )
    num_test_samples = len(test_df.index)

    valid_generator = MyGenerator(
        image_filenames=valid_df['file'].tolist(),
        targets=valid_df,
        resize_target=target_shape,
        batch_size=batch_size,
        base_path=base_path
    )
    num_valid_samples = len(valid_df.index)

    print('Dataset summary:')
    print(df.count())
    print('Train samples:%d' % num_train_samples)
    print('Valid samples:%d' % num_valid_samples)
    print('Test samples:%d' % num_test_samples)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

    # build model
    model = build_model(input_shape)

    # compile the model
    model.compile(optimizer=sgd(lr=learning_rate, momentum=momentum),
                  loss=root_mean_squared_error,
                  metrics=[R2]
                  )

    model.summary()

    # callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                 ModelCheckpoint(filepath='./training_outputs/weights_checkpoint_' + timestamp + '.h5', monitor='val_loss',
                                 verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)]

    # train the model
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=(num_train_samples // batch_size),
        epochs=500,
        callbacks=callbacks,
        validation_data=valid_generator,
        validation_steps=(num_valid_samples // batch_size),
        verbose=1
    )

    print('Training ended...')

    plt.plot(history.history['R2'], label='Train R2', color='red')
    plt.plot(history.history['val_R2'], label='Valid R2', color='green')
    plt.title('model R2 over epochs')
    plt.ylabel('R2')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./training_outputs/r2[' + timestamp + '].png')

    plt.figure()
    plt.plot(history.history['loss'], label='Train loss', color='red')
    plt.plot(history.history['val_loss'], label='Valid loss', color='green')
    plt.title('model loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./training_outputs/loss[' + timestamp + '].png')

    # test_score = model.evaluate(testImages, testY)
    test_score = model.evaluate_generator(
        generator=test_generator,
        steps=(num_test_samples // batch_size),
        verbose=1
    )
    # performannce on test set
    print('#######################')
    print('Performance on test set')
    print('%s: %.2f%%' % (model.metrics_names[1], test_score[1] * 100))
    print('%s: %.2f%%' % (model.metrics_names[0], test_score[0]))

    # serialize model to JSON
    model_json = model.to_json()
    with open("./training_outputs/tool_" + timestamp + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./training_outputs/tool_" + timestamp + ".h5")
    print("Saved model to disk")

    start = time.time()
    preds = model.predict_generator(
        generator=test_generator
    )
    end = time.time()
    print('Predictions a total of %.3f and an average of %.3f for each frame.' % (end - start, (end-start)/num_test_samples))

    i = 0
    # load filenames and ground truth
    testX = test_df['file'].tolist()
    testY = load_targets(test_df)

    for p, filename, t in zip(preds, testX, testY):
        im = cv.imread(base_path + filename)
        im = im.astype(np.float32)
        # draw prediction
        cv.circle(im, (int(p[0] * 240), int(p[1] * 320)), 1, (0, 0, 255), -1, cv.LINE_AA)
        # draw ground truth
        cv.circle(im, (int(t[0]), int(t[1])), 1, (0, 255, 0), -1, cv.LINE_AA)
        cv.putText(im, 'pred:(' + str(p[0] * 240) + ',' + str(p[1] * 320) + ')', (20, 220),
                   cv.FONT_HERSHEY_PLAIN, .6,
                   color=(0, 0, 255))
        cv.putText(im, 'real:(' + str(t[0]) + ',' + str(t[1]) + ')', (20, 200),
                   cv.FONT_HERSHEY_PLAIN, .6,
                   color=(0, 255, 0))

        cv.imwrite('./preds/pred%d.png' % i, im)

        i += 1

    diff = np.abs(testY - preds)
    error = []
    for n in diff:
        error.append(np.sqrt(n[0]**2 + n[1]**2))
    avg_error = np.mean(error)
    print(avg_error)
    std_error = np.std(error)
    print(std_error)
    errors = pd.DataFrame(zip(testX, testY, preds, error),
                          columns=['file', 'real', 'predicted', 'error'])

    labels = errors['file'].to_list()
    labels = [l[:-4] for l in labels]
    plt.figure()
    plt.title('Error distribution - Avg: %.2f - Std: %.2f' % (avg_error, std_error))
    plt.scatter(labels, errors['error'].to_list())
    plt.xticks(rotation=90, fontsize=6)
    plt.savefig('./training_outputs/errors_distr[' + timestamp + '].png')

    with open("./training_outputs/params_" + timestamp + ".txt", "w") as text_file:
        text_file.write("Training params:\nbatch_size=%d\nlearning_rate=%.3f\nmomentum=%.2f"
                        % (batch_size, learning_rate, momentum))


if __name__ == '__main__':
    main()
