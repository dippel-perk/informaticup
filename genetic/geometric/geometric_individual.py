from genetic.image_individual import ImageIndividual
from utils.image_utilities import ImageUtilities
from PIL import Image


class GeometricIndividual(ImageIndividual):
    def __init__(self, geometric_objects):
        self._geometric_objects = geometric_objects
        img = self._drawImage()
        super().__init__(image=img)
        self.classification = None

    def _drawImage(self):
        img = Image.open('road_resized.jpg')
        for object in self._geometric_objects:
            object.draw(img)
        return img

    def get_objects(self):
        return self._geometric_objects
