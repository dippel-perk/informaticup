from PIL.Image import Image
from typing import List
from classifier.classification import ImageClassification

class Classifier:
    """
    The non-functional base class for a image classifier.
    """

    DESIRED_IMAGE_EXTENSION = "PNG"
    DESIRED_IMAGE_WIDTH = 64
    DESIRED_IMAGE_HEIGHT = 64

    DESIRED_IMAGE_DIMENSIONS = (DESIRED_IMAGE_WIDTH, DESIRED_IMAGE_HEIGHT)

    def classify(self, image: Image) -> ImageClassification:
        """
        This method should classify a single image and return the classification.
        :param image: The input image.
        :return: The classification result.
        """
        raise NotImplementedError

    def classify_batch(self, images: List[Image]) -> ImageClassification:
        """
        This method should classify a list of images and return the classifications.
        :param images: The input images.
        :return: The list of classification results.
        """
        raise NotImplementedError
