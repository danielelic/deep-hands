from __future__ import print_function

import keras.preprocessing.image
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Convolution2D, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2

from data import load_train_data, load_test_data

# input image dimensions
img_rows, img_cols = 80, 80

num_classes = 3
channels = 3
weight_decay = 0.0005


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def getcaffenet():
    input_shape = (img_rows, img_cols, channels)

    model = Sequential()

    # Conv1
    model.add(Convolution2D(nb_filter=96, nb_row=11, nb_col=11, border_mode='valid', input_shape=input_shape
                            , subsample=(4, 4),
                            W_regularizer=l2(weight_decay)))  # subsample is stride
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Conv2
    model.add(
        Convolution2D(256, 5, 5, border_mode='same', W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Conv3
    model.add(
        Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))

    # Conv4
    model.add(
        Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))

    # Conv5
    model.add(
        Convolution2D(256, 3, 3, border_mode='same', W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # Fc6
    model.add(Dense(4096, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fc7
    model.add(Dense(4096, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fc8
    model.add(Dense(num_classes, W_regularizer=l2(weight_decay)))
    model.add(Activation('softmax'))

    # optimizer=SGD
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', precision, recall, f1score])

    return model


if __name__ == '__main__':
    x_train, y_train, train_ids = load_train_data()
    x_test, y_test, test_ids = load_test_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

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
    model = getcaffenet()

    csv_logger = CSVLogger('log-caffenet.csv')
    model_checkpoint = ModelCheckpoint('weights-caffenet.h5', monitor='acc', save_best_only=True)

    gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15.,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.4,
        zoom_range=0.4,
        channel_shift_range=0.5,
        horizontal_flip=True,
        vertical_flip=False
    )

    batch_size = 32
    train_steps = int(x_train.shape[0] / batch_size) + 1
    validation_steps = int(x_test.shape[0] / 32) + 1
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    model.summary()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                        steps_per_epoch=train_steps * 10,
                        epochs=200, verbose=1,
                        validation_data=gen.flow(x_test, y_test, batch_size=32, shuffle=False),
                        validation_steps=validation_steps,
                        callbacks=[csv_logger, model_checkpoint, reduce_lr])

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
