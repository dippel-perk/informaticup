import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import Sequential
from keras.callbacks import ModelCheckpoint


class Classifier:

    def predict(self, X):
        pass


class OfflineClassifier(Classifier):

    def __init__(self, img_size=64, num_classes=43, weights_file=None):
        self._img_size = img_size
        self._num_classes = num_classes

        self._model = self._build_model()
        self._model.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        if not weights_file is None:
            self._model.load_weights(weights_file)

    def train(self, train_path, test_image_path, test_df):
        datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = datagen.flow_from_directory(
            train_path,
            target_size=(self._img_size, self._img_size),
            batch_size=32,
            class_mode='categorical')
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df, directory=test_image_path,
            x_col='Filename', y_col='ClassId',
            target_size=(self._img_size, self._img_size),
            batch_size=32,
            class_mode='categorical')

        filepath = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
        model_checkpoint_cb = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                              mode='auto')

        self._model.fit_generator(train_generator, epochs=20, callbacks=[model_checkpoint_cb],
                                  validation_data=test_generator)

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
