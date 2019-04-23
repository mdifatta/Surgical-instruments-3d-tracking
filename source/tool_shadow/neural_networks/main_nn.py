from keras.optimizers import rmsprop
from keras.losses import binary_crossentropy
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pandas as pd
from matplotlib import pyplot as plt

from neural_networks.models import mobile_net


def main():
    # params
    batch_size = 32
    target_shape = (320, 240)
    learning_rate = .002
    input_shape = (320, 240, 3)

    # read train csv
    # ######################## local ############################
    train_path = '../../data/datasets/all_distance_frames/'
    train = pd.read_csv('../../data/targets/train.csv', sep=';')
    # ##########################################################

    # ################# for colab ####################
    #train_path = './frames/'
    #train = pd.read_csv('./targets/train.csv', sep=';')
    # ################################################
    train['valid'] = train['valid'].astype('str')
    train = train.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # read test csv
    # ######################## local ############################
    test_path = '../../data/datasets/all_distance_frames/'
    test = pd.read_csv('../../data/targets/test.csv', sep=';')
    # ##########################################################

    # ################# for colab ####################
    #test_path = './frames/'
    #test = pd.read_csv('./targets/test.csv', sep=';')
    # ################################################
    test['valid'] = test['valid'].astype('str')
    test = test.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # read valid csv
    # ######################## local ############################
    valid_path = '../../data/datasets/all_distance_frames/'
    valid = pd.read_csv('../../data/targets/valid.csv', sep=';')
    # ##########################################################

    # ################# for colab ####################
    #valid_path = './frames/'
    #valid = pd.read_csv('./targets/valid.csv', sep=';')
    # ################################################
    valid['valid'] = valid['valid'].astype('str')
    valid = valid.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # create training set generator
    train_data_gen = ImageDataGenerator()
    train_generator = train_data_gen.flow_from_dataframe(
        dataframe=train,
        directory=train_path,
        x_col='file',
        y_col='valid',
        class_mode='categorical',
        classes=['0', '1'],
        target_size=target_shape,
        batch_size=batch_size,
        color_mode='rgb'
    )

    # create validation set generator
    valid_data_gen = ImageDataGenerator()
    valid_generator = valid_data_gen.flow_from_dataframe(
        dataframe=valid,
        directory=valid_path,
        x_col='file',
        y_col='valid',
        class_mode='categorical',
        classes=['0', '1'],
        target_size=target_shape,
        batch_size=batch_size,
        color_mode='rgb'
    )

    # create test set generator
    test_data_gen = ImageDataGenerator()
    test_generator = test_data_gen.flow_from_dataframe(
        dataframe=test,
        directory=test_path,
        x_col='file',
        y_col='valid',
        class_mode='categorical',
        classes=['0', '1'],
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
        validation_steps=STEP_SIZE_VALID,
        epochs=1000,
        callbacks=callbacks,
        verbose=1
    )

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy over epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train', 'valid'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['vall_loss'])
    plt.title('model loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'valid'])
    plt.show()

    test_score = model.evaluate_generator(generator=test_generator,
                                          verbose=0)

    print(test_score)


if __name__ == '__main__':
    main()

# TODO: implement test
# TODO: try to use DataGenerator

