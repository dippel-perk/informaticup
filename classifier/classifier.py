import string
from PIL.Image import Image
from typing import List

class Classifier():

    DESIRED_IMAGE_EXTENSION = "PNG"
    DESIRED_IMAGE_WIDTH = 64
    DESIRED_IMAGE_HEIGHT = 64

    DESIRED_IMAGE_DIMENSIONS = (DESIRED_IMAGE_WIDTH, DESIRED_IMAGE_HEIGHT)

    def classify(self, image: Image):
       raise NotImplementedError

    def classify_batch(self, images: List[Image]):
        raise NotImplementedError



