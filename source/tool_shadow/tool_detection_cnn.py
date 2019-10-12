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
import seaborn as sns
from keras import backend as K
from keras.activations import relu, sigmoid
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import Sequence
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# custom DataGenerator used to feed the CNN with data batches
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


# original CNN topology from the paper with addition of Ridge regularization
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation=relu, padding='same',
                     data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.05))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.1))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=250, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=250, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=2, activation=relu, kernel_regularizer=l2(5e-4)))
    # model.add(LeakyReLU(alpha=.2))

    return model


# CNN topology with more filters
def build_wider_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation=relu, padding='same',
                     data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.05))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.1))
    model.add(Conv2D(filters=128, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=250, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=250, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=2, activation=relu, kernel_regularizer=l2(5e-4)))

    return model


# CNN topology with more filter and more layers
def build_deeper_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation=relu, padding='same',
                     data_format='channels_last', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.05))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.1))
    model.add(Conv2D(filters=128, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=256, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=512, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=512, kernel_size=(2, 2), activation=relu, padding='same', kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=350, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=350, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=350, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=350, activation=relu, kernel_regularizer=l2(5e-4)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=2, activation=relu, kernel_regularizer=l2(5e-4)))

    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# custom loss implementing Euclidean Distance
def custom_loss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis = -1, keepdims = True)))


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
    # construct the argument parser and parse the cmd line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env", type=str, required=True,
                    help="system where to execute")
    args = vars(ap.parse_args())

    # ######################### HYPER-PARAMS ##############################
    batch_size = 64
    target_shape = (320, 240)
    learning_rate = .001
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
    elif ENV == 'colab':
        # ################# for colab ####################
        # create base path for images
        base_path = './frames/'
        # read targets csv
        df = pd.read_csv('./targets/targets.csv', sep=';')
        # take only valid frames
        df = df[df['p'] != '(-1, -1)']
        # shuffle data
        df = shuffle(df)
    else:
        # ######################## computer lab ############################
        # create base path for images
        base_path = '../../data/datasets/all_distance_frames/'
        # read targets csv
        df = pd.read_csv('../../data/targets/targets-v2.csv', sep=';')
        # take only valid frames
        df = df[df['p'] != '(-1, -1)']
        # shuffle data
        df = shuffle(df)

    # split train and test as 90/10
    (train_df, test_df) = train_test_split(df,
                                           test_size=.1,
                                           train_size=.9)

    # split train and valid as 80/20 of previous train
    (train_df, valid_df) = train_test_split(train_df,
                                            test_size=.2,
                                            train_size=.8)

    # create data generators for train, test and valid set
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

    # print a summary of dataset size and split size
    print('Dataset summary:')
    print(df.count())
    print('Train samples:%d' % num_train_samples)
    print('Valid samples:%d' % num_valid_samples)
    print('Test samples:%d' % num_test_samples)

    # create and print a timestamp of when the training is starting; it's also used for file naming
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    print('####################################')
    print('Training starting at %s.' % timestamp)

    # build model
    model = build_deeper_model(input_shape)

    # compile the model. i.e. define optimizer, loss function and additional metrics
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=custom_loss,
                  metrics=[R2]
                  )

    # print model's summary
    model.summary()

    # callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                 ModelCheckpoint(filepath='./training_outputs/weights_checkpoint_{}.h5'.format(timestamp),
                                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min',
                                 period=1)]

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

    # plot and save metrics' evolution
    plt.plot(history.history['R2'], label='Train R2', color='red')
    plt.plot(history.history['val_R2'], label='Valid R2', color='green')
    plt.title('model R2 over epochs')
    plt.ylabel('R2')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./training_outputs/r2-{}.png'.format(timestamp))

    # plot and save loss' evolution
    plt.figure()
    plt.plot(history.history['loss'], label='Train loss', color='red')
    plt.plot(history.history['val_loss'], label='Valid loss', color='green')
    plt.title('model loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./training_outputs/loss-{}.png'.format(timestamp))

    # test_score = model.evaluate(testImages, testY)
    # perform test onf the test set
    test_score = model.evaluate_generator(
        generator=test_generator,
        steps=(num_test_samples // batch_size),
        verbose=1
    )
    # print performance on test set
    print('#######################')
    print('Performance on test set')
    print('%s: %.2f%%' % (model.metrics_names[1], test_score[1] * 100))
    print('%s: %.2f' % (model.metrics_names[0], test_score[0]))

    # serialize model to JSON
    model_json = model.to_json()
    with open("./training_outputs/tool_{}.json".format(timestamp), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./training_outputs/tool_{}.h5".format(timestamp))
    print("Saved model to disk")

    # perform predictions on the test set again to print predictions and compute running time
    start = time.time()
    preds = model.predict_generator(
        generator=test_generator
    )
    end = time.time()
    print('Predictions a total of %.3f and an average of %.3f for each frame.' % (end - start, (end-start)/num_test_samples))

    # load filenames and ground truth
    testX = test_df['file'].tolist()
    testY = load_targets(test_df)

    # rescale predicted coordinates
    for p in preds:
        p[0] = p[0] * 240
        p[1] = p[1] * 320

    # generate predictions files
    error = []
    for p, filename, t in zip(preds, testX, testY):
        error.append(distance.euclidean(p, t))
        im = cv.imread(base_path + filename)
        im = im.astype(np.float32)
        # draw prediction
        cv.circle(im, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1, cv.LINE_AA)
        # draw ground truth
        cv.circle(im, (int(t[0]), int(t[1])), 1, (0, 255, 0), -1, cv.LINE_AA)
        cv.putText(im, 'pred:(' + str(p[0]) + ',' + str(p[1]) + ')', (20, 220),
                   cv.FONT_HERSHEY_PLAIN, .6,
                   color=(0, 0, 255))
        cv.putText(im, 'real:(' + str(t[0]) + ',' + str(t[1]) + ')', (20, 200),
                   cv.FONT_HERSHEY_PLAIN, .6,
                   color=(0, 255, 0))

        cv.imwrite('./preds/%s' % filename, im)

    # compute summary statistics and save plots
    avg_error = float(np.mean(error, dtype=np.float64))
    print(avg_error)
    std_error = float(np.std(error, dtype=np.float64))
    print(std_error)
    errors = pd.DataFrame(zip(testX, testY, preds, error),
                          columns=['file', 'real', 'predicted', 'error'])
    errors[['file', 'error']].to_csv("./training_outputs/error-on-frames-{}.txt".format(timestamp))

    labels = errors['file'].to_list()
    labels = [l[:-4] for l in labels]
    plt.figure()
    plt.title('Error distribution - Avg: {} - Std: {}}'.format(avg_error, std_error))
    plt.scatter(labels, errors['error'].to_list())
    plt.xticks(rotation=90, fontsize=6)
    plt.savefig('./training_outputs/errors_distr-{}.png'.format(timestamp))

    with open("./training_outputs/model_snapshot_%s.txt" % timestamp, "w") as text_file:
        text_file.write("Training params:\nbatch_size=%d\nlearning_rate=%.3f\n"
                        % (batch_size, learning_rate, momentum))


if __name__ == '__main__':
    main()
