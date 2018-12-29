from classifier.neural_net import NeuralNet
from PIL import Image
from classifier.classifier import Classifier
from classifier.classification import Class, ImageClassification


class OfflineClassifier(Classifier):
    def __init__(self, weights_file='weights-20-0.98.hdf5'):
        self._neural_net = NeuralNet(weights_file=weights_file)

    def classify(self, file_name):
        prediction = [(class_id, probability) for class_id, probability in enumerate(self._neural_net.predict(file_name).tolist()[0][:43])]
        prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
        classes = [Class(str(class_id), probability) for class_id, probability in prediction]
        return ImageClassification(file_name, classes)
