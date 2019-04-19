from keras.optimizers import rmsprop
from keras.losses import binary_crossentropy
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pandas as pd

from neural_networks.models import mobile_net


def main():
    # params
    batch_size = 32
    target_shape = (640, 480)
    learning_rate = .002
    input_shape = (640, 480, 3)

    # TODO: change paths
    # TODO: create a unique csv file to shuffle, split in train and test and read
    # read train csv
    train_path = '../../data/datasets/2d_frames_folders/prova1/'
    train = pd.read_csv('../../data/targets/prova_train.csv', sep=';')
    train['valid'] = train['valid'].astype('str')
    train = train.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # read test csv
    test_path = '../../data/datasets/2d_frames_folders/prova1/'
    test = pd.read_csv('../../data/targets/prova_test.csv', sep=';')
    test['valid'] = test['valid'].astype('str')
    test = test.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # create training set generator
    train_data_gen = ImageDataGenerator(validation_split=0.25)
    train_generator = train_data_gen.flow_from_dataframe(
        dataframe=train,
        directory=train_path,
        x_col='file',
        y_col='valid',
        class_mode='binary',
        target_size=target_shape,
        batch_size=batch_size,
        color_mode='rgb',
        subset='training'
    )

    # create validation set generator
    valid_generator = train_data_gen.flow_from_dataframe(
        dataframe=train,
        directory=train_path,
        x_col='file',
        y_col='valid',
        class_mode='binary',
        target_size=target_shape,
        batch_size=batch_size,
        color_mode='rgb',
        subset='validation'
    )

    # create test set generator
    test_data_gen = ImageDataGenerator()
    test_generator = test_data_gen.flow_from_dataframe(
        dataframe=test,
        directory=test_path,
        x_col='file',
        y_col='valid',
        class_mode='binary',
        target_size=target_shape,
        color_mode='rgb'
    )

    # build the model
    model = mobile_net(input_shape)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    # compile the model
    model.compile(optimizer=rmsprop(lr=learning_rate),
                  loss=binary_crossentropy,
                  metrics=['accuracy']
                  )

    model.summary()

    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

    # train the model
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=50,
        epochs=1000,
        callbacks=callbacks,
        verbose=0
    )


if __name__ == '__main__':
    main()

# TODO: implement test
# TODO: augmentation functions
# TODO: try to use DataGenerator

