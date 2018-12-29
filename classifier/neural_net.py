from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
from PIL import Image
import pandas as pd
import os.path


class NeuralNet:
    def __init__(self, img_size=64, num_classes=44, weights_file=None):
        self._img_size = img_size
        self._num_classes = num_classes

        self._model = self._build_model()
        self._model.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        if not weights_file is None:
            self._model.load_weights(weights_file)

    def get_model(self):
        return self._model

    def train(self, path, validation_split=0.2, epochs=20, batch_size=128):
        datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)

        train_generator = datagen.flow_from_directory(
            path,
            target_size=(self._img_size, self._img_size),
            batch_size=batch_size,
            classes=[str(i).zfill(5) for i in range(self._num_classes)],
            class_mode='categorical',
            subset='training')

        validation_generator = datagen.flow_from_directory(
            path,
            target_size=(self._img_size, self._img_size),
            batch_size=batch_size,
            classes=[str(i).zfill(5) for i in range(self._num_classes)],
            class_mode='categorical',
            subset='validation')

        filepath = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
        model_checkpoint_cb = ModelCheckpoint(filepath=filepath, monitor='acc', verbose=1, save_best_only=True,
                                              mode='auto')

        self._model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=train_generator.samples,
                                  callbacks=[model_checkpoint_cb], validation_data=validation_generator,
                                  validation_steps=validation_generator.samples)

    def predict(self, file_name):
        img = Image.open(file_name)
        x = np.array(img, dtype='float32')
        x /= 255
        x = x[np.newaxis, :, :, :]
        result = self._model.predict(x, batch_size=1)
        return result

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(self._img_size, self._img_size, 3),
                         activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self._num_classes, activation='softmax'))
        return model


if __name__ == '__main__':
    net = NeuralNet()
    prefix = '../../GTSRB/'
    test_df = pd.read_csv(prefix + 'Final_Test/test.csv', delimiter=';')
    net.train(os.path.join(prefix, 'Final_Training/Images'))
