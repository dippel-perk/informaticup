from typing import List

from PIL import Image

from config.classifier_configuration import ClassifierConfiguration
from config.geometric_individual_configuration import GeometricIndividualConfiguration
from genetic.geometric.geometric_objects import GeometricObject
from genetic.image_individual import ImageIndividual


class GeometricIndividual(ImageIndividual):
    """
    Represents a geometric individual. A geometric individual is an image individual which contains additional
    information about geometric objects which are part of the image.
    """

    def __init__(self, geometric_objects: List[GeometricObject]):
        self._geometric_objects = geometric_objects
        super().__init__(image=self._drawImage())

    def _drawImage(self) -> Image:
        """
        Draws an image which contains all geometric objects.
        :return: The generated image
        """
        img = Image.new('RGB', GeometricIndividualConfiguration.IMAGE_DIMENSION)
        for object in self._geometric_objects:
            object.draw(img)
        img = img.resize(ClassifierConfiguration.DESIRED_IMAGE_DIMENSIONS, resample=Image.ANTIALIAS)
        return img

    def get_objects(self) -> List[GeometricObject]:
        return self._geometric_objects
