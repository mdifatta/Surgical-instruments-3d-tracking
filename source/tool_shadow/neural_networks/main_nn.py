import pandas as pd
from keras.applications.densenet import DenseNet121
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def main():
    # ######################### PARAMS ##############################
    batch_size = 32
    target_shape = (320, 240)
    learning_rate = .0001
    input_shape = (320, 240, 3)
    ENV = 'local'
    # ###############################################################

    if ENV == 'local':
        # ######################## local ############################
        # read train csv
        train_path = '../../../data/datasets/all_distance_frames/'
        train = pd.read_csv('../../../data/targets/train.csv', sep=';')

        # read test csv
        test_path = '../../../data/datasets/all_distance_frames/'
        test = pd.read_csv('../../../data/targets/test.csv', sep=';')

        # read valid csv
        valid_path = '../../../data/datasets/all_distance_frames/'
        valid = pd.read_csv('../../../data/targets/valid.csv', sep=';')
    else:
        # ################# for colab ####################
        # read train csv
        train_path = './frames/'
        train = pd.read_csv('./targets/train.csv', sep=';')

        # read test csv
        test_path = './frames/'
        test = pd.read_csv('./targets/test.csv', sep=';')

        # read valid csv
        valid_path = './frames/'
        valid = pd.read_csv('./targets/valid.csv', sep=';')

    train['valid'] = train['valid'].astype('str')
    train = train.drop(labels=['p1', 'p2', 'dist'], axis=1)

    test['valid'] = test['valid'].astype('str')
    test = test.drop(labels=['p1', 'p2', 'dist'], axis=1)

    valid['valid'] = valid['valid'].astype('str')
    valid = valid.drop(labels=['p1', 'p2', 'dist'], axis=1)

    # dataset's balance
    val = train[train.valid == '1']['valid'].count()
    inval = train[train.valid == '0']['valid'].count()
    tot = val + inval
    print('Training set: %d (%.2f) VALID, %d (%.2f) INVALID' %
          (val, val/tot, inval, inval/tot))

    val = test[test.valid == '1']['valid'].count()
    inval = test[test.valid == '0']['valid'].count()
    tot = val + inval
    print('Test set: %d (%.2f) VALID, %d (%.2f) INVALID' %
          (val, val/tot, inval, inval/tot))

    val = valid[valid.valid == '1']['valid'].count()
    inval = valid[valid.valid == '0']['valid'].count()
    tot = val + inval
    print('Validation set: %d (%.2f) VALID, %d (%.2f) INVALID' %
          (val, val/tot, inval, inval/tot))

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
    # model = mobile_net(input_shape)
    # model = MobileNet(input_shape=input_shape, weights=None, classes=2)
    # model = MobileNetV2(input_shape=input_shape, weights=None, classes=2)
    model = DenseNet121(include_top=False, input_shape=input_shape, weights='None')

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    # compile the model
    model.compile(optimizer=rmsprop(lr=learning_rate),
                  loss=categorical_crossentropy,
                  metrics=['acc']
                  )

    model.summary()

    # callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

    class_weights = {0: 1.1,
                     1: 1.0
                     }

    # train the model
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=1000,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )

    print('Training ended...')

    plt.plot(history.history['acc'], label='Train acc', color='red')
    plt.plot(history.history['val_acc'], label='Valid acc', color='green')
    plt.title('model accuracy over epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./accuracy['+ str(batch_size) +','+ str(learning_rate) +'].png')

    plt.figure()
    plt.plot(history.history['loss'], label='Train loss', color='red')
    plt.plot(history.history['val_loss'], label='Valid loss', color='green')
    plt.title('model loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./loss['+ str(batch_size) +','+ str(learning_rate) +'].png')

    test_score = model.evaluate_generator(generator=test_generator,
                                          steps=STEP_SIZE_TEST,
                                          verbose=1)

    print('%s: %.2f%%' % (model.metrics_names[1], test_score[1] * 100))
    print('%s: %.2f%%' % (model.metrics_names[0], test_score[0]))

    # serialize model to JSON
    model_json = model.to_json()
    with open("./mobilev2_b"+str(batch_size)+"_lr"+str(learning_rate)+"ds.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./mobilev2_b"+str(batch_size)+"_lr"+str(learning_rate)+"ds.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()


