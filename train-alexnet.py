from __future__ import print_function

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

from data import load_train_data, load_test_data

# input image dimensions
img_rows, img_cols = 80, 80

num_classes = 3


def getAlexNet(heatmap=False):
    model = Sequential()

    # Conv layer 1 output shape (55, 55, 48)
    model.add(Conv2D(
        kernel_size=(11, 11),
        data_format="channels_last",
        activation="relu",
        filters=48,
        strides=(4, 4),
        input_shape=input_shape
    ))
    model.add(Dropout(0.25))

    # Conv layer 2 output shape (27, 27, 128)
    model.add(Conv2D(
        strides=(2, 2),
        kernel_size=(5, 5),
        activation="relu",
        filters=128
    ))
    model.add(Dropout(0.25))

    # Conv layer 3 output shape (13, 13, 192)
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=192,
        padding="same",
        strides=(2, 2)
    ))
    model.add(Dropout(0.25))

    # Conv layer 4 output shape (13, 13, 192)
    model.add(Conv2D(
        padding="same",
        activation="relu",
        kernel_size=(3, 3),
        filters=192
    ))
    model.add(Dropout(0.25))

    # Conv layer 5 output shape (128, 13, 13)
    model.add(Conv2D(
        padding="same",
        activation="relu",
        kernel_size=(3, 3),
        filters=128
    ))
    model.add(Dropout(0.25))

    # fully connected layer 1
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))

    # fully connected layer 2
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))

    # output
    model.add(Dense(num_classes, activation='softmax'))

    # optimizer=SGD
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    x_train, y_train, train_ids = load_train_data()
    x_test, y_test, test_ids = load_test_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = getAlexNet()

    csv_logger = CSVLogger('log-alexnet.csv')
    model_checkpoint = ModelCheckpoint('weights-alexnet.h5', monitor='accuracy', save_best_only=True)

    model.summary()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, callbacks=[csv_logger, model_checkpoint])
