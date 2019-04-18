from keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Dense, SeparableConv2D
from keras.models import Sequential


def mobile_net(input_shape, alpha=1):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=512, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=512, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=512, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=512, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=512, kernel_size=(3, 3), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', depth_multiplier=alpha))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(7, 7)))
    model.add(Dense(units=2, activation='softmax'))

    return model
