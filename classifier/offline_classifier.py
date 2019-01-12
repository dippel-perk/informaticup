from classifier.neural_net import NeuralNet
from PIL.Image import Image
from classifier.classifier import Classifier
from classifier.classification import Class, ImageClassification
from typing import List


class OfflineClassifier(Classifier):
    """
    Offline classifier calls the substitute network to classify given images.
    """
    def __init__(self, weights_file='weights/weights-partial.hdf5'):
        self._neural_net = NeuralNet(weights_file=weights_file)

    def classify(self, image: Image):
        """
        Classifies a single image with the substitute network and returns the classification.
        :param image: The input image.
        :return: The classification result.
        """
        return self.classify_batch([image])[0]

    def classify_batch(self, images: List[Image]):
        """
        Classifies a list of images with the substitute network and returns the classifications.
        :param images: The input images.
        :return: The list of classification results.
        """
        classifications = []
        for prediction in self._neural_net.predict(images).tolist():
            classes = [Class(str(class_id), probability) for class_id, probability in enumerate(prediction)]
            classifications.append(ImageClassification(classes))
        return classifications

    def __repr__(self):
        return 'OfflineClassifier'
