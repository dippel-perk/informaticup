import random as rd
from typing import List

from PIL.Image import Image

from classifier.classification import ImageClassification, Class
from classifier.classifier import Classifier
from utils.road_sign_class_mapper_utilities import RoadSignClassMapperUtilities


class ClassifierMock(Classifier):
    def _get_random_classification(self):
        classes = [Class(name=RoadSignClassMapperUtilities.get_name_by_class(i),
                         confidence=rd.random()) for i in range(5)]
        return ImageClassification(classes=classes)

    def classify(self, image: Image):
        return self._get_random_classification()

    def classify_batch(self, images: List[Image]):
        return [self._get_random_classification() for i in images]
