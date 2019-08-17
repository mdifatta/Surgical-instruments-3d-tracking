import argparse
import datetime
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from keras.activations import relu, tanh
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.optimizers import sgd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
    model.add(Dense(units=2, activation=tanh))

    return model


def load_images(df: pd.DataFrame, basePath, shape):
    print('Loading images ...')
    # images container
    images = []
    df = shuffle(df)

    for index, row in tqdm.tqdm(df.iterrows()):
        filename = row['file']
        img = cv.imread(os.path.join(basePath, filename), cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, shape)
        img = np.expand_dims(img, axis=2)

        images.append(img)

    print('Images loaded.')

    return np.array(images)


def load_targets(df: pd.DataFrame):
    print('Loading targets ...')
    targets = []
    df = shuffle(df)
    df.loc[df['p2'] == '-1', 'p2'] = '(-1, -1)'
    df['p'] = [eval(x) for x in df['p2']]
    for x in df['p']:
        targets.append(np.array(x))
    df = df.drop(labels=['p1', 'p2', 'valid', 'dist'], axis=1)
    print('Targets loaded.')
    return np.asarray(targets, dtype=np.float64)


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env", type=str, required=True,
                    help="system where to execute")
    args = vars(ap.parse_args())

    # ######################### PARAMS ##############################
    batch_size = 32
    target_shape = (320, 240)
    learning_rate = .01
    momentum = .9
    input_shape = (240, 320, 1)
    ENV = args['env']
    # ###############################################################

    if ENV == 'local':
        # ######################## local ############################
        # create base path for images
        base_path = '../../data/datasets/all_distance_frames/'
        # read targets csv
        df = pd.read_csv('../../data/targets/targets.csv', sep=';')
    else:
        # ################# for colab ####################
        # create base path for images
        base_path = './frames/'
        # read targets csv
        df = pd.read_csv('./targets/targets.csv', sep=';')

    images = load_images(df, base_path, target_shape)
    images = images / 255.0

    targets = load_targets(df)
    for i in range(len(targets)):
        if np.all(targets[i] > 0.0):
            targets[i] = targets[i] / 255

    # split train and test as 80/20
    (trainY, testY, trainImages, testImages) = train_test_split(targets,
                                                                images,
                                                                test_size=.2,
                                                                train_size=.8,
                                                                random_state=948)

    # split train and valid as 80/20 of previous test
    (trainY, validY, trainImages, validImages) = train_test_split(trainY,
                                                                  trainImages,
                                                                  test_size=.1,
                                                                  train_size=.9,
                                                                  random_state=36)

    # build model
    model = build_model(input_shape)

    # compile the model
    model.compile(optimizer=sgd(lr=learning_rate, momentum=momentum),
                  loss=mean_squared_error,
                  metrics=['mse']
                  )

    model.summary()

    # callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=20, mode='min')]

    # train the model
    history = model.fit(
        trainImages,
        trainY,
        batch_size=batch_size,
        epochs=500,
        callbacks=callbacks,
        validation_data=(validImages, validY),
        verbose=1
    )

    print('Training ended...')

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

    plt.plot(history.history['mean_squared_error'], label='Train mse', color='red')
    plt.plot(history.history['val_mean_squared_error'], label='Valid mse', color='green')
    plt.title('model mse over epochs')
    plt.ylabel('mse')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./mse[' + timestamp + '].png')

    plt.figure()
    plt.plot(history.history['loss'], label='Train loss', color='red')
    plt.plot(history.history['val_loss'], label='Valid loss', color='green')
    plt.title('model loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./loss[' + timestamp + '].png')

    test_score = model.evaluate(testImages, testY)

    print('%s: %.2f%%' % (model.metrics_names[1], test_score[1] * 100))
    print('%s: %.2f%%' % (model.metrics_names[0], test_score[0]))

    # serialize model to JSON
    model_json = model.to_json()
    with open("./shadow_" + timestamp + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./shadow_" + timestamp + ".h5")
    print("Saved model to disk")

    preds = model.predict(testImages)
    i = 0
    for p, im, t in zip(preds, testImages, testY):
        c = (im * 255.0).astype(np.float32)
        c = cv.cvtColor(c, cv.COLOR_GRAY2BGR)
        cv.circle(c, (p[0], p[1]), 3, (0, 0, 255), -1, cv.LINE_AA)

        cv.imwrite('./preds/pred%d.png' % i, c)

        i += 1


if __name__ == '__main__':
    main()
