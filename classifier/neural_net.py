import os.path
from typing import List

import numpy as np
from PIL.Image import Image
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NeuralNet:
    """
    A convolutional neural network that can classify squared images.
    """

    def __init__(self, img_size: int = 64, num_classes: int = 43, weights_file: str = None):
        """
        Constructs a new Neural Network. If the weights file is given, the initial weights are loaded from the file.
        :param img_size: The size of the images that should be classified (only quadratic images)
        :param num_classes: The number of possible classes
        :param weights_file: A .hdf5 file that contains weights for the network
        """
        self._img_size = img_size
        self._num_classes = num_classes

        self._model = self._build_model()
        self._model.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        if not weights_file is None:
            self._model.load_weights(weights_file)

    def get_model(self):
        """
        Returns the undelying Keras model
        :return: Keras model of the the neural network
        """
        return self._model

    def train(self, path: str, validation_split: float = 0.2, epochs: int = 20, batch_size: int = 128):
        """
        Trains the neural network
        :param path: The folder where the images are located. The folder must contain for each class a folder with the images.
        :param validation_split: percentage of the data that should be used for validation while training.
        :param epochs: How many epochs the network should be trained.
        :param batch_size: The training batch size.
        """
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
        model_checkpoint_cb = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                              mode='auto')

        self._model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=train_generator.samples / batch_size,
                                  callbacks=[model_checkpoint_cb], validation_data=validation_generator,
                                  validation_steps=validation_generator.samples / batch_size)

    def predict(self, images: List[Image]):
        """
        Computes the output of the neural network for every image in a single pass.
        :param images: A list of `n` images that should be classified.
        :return: A `n * classes` matrix with the prediction results.
        """
        batch = []
        for img in images:
            batch.append(np.array(img, dtype='float32') / 255)
        result = self._model.predict(np.array(batch), batch_size=len(images))
        return result

    def _build_model(self):
        """
        Constructs the Keras model of the neural network.
        :return: Keras model
        """
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
