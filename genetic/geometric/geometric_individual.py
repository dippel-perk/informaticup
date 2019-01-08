from genetic.image_individual import ImageIndividual
from utils.image_utilities import ImageUtilities
from PIL import Image

IMAGE_DIMENSION = 1024


class GeometricIndividual(ImageIndividual):
    def __init__(self, geometric_objects):
        self._geometric_objects = geometric_objects
        img = self._drawImage()
        super().__init__(image=img)
        self.classification = None

    def _drawImage(self):
        #img = Image.open('road_resized.jpg')
        img = Image.new('RGB', (IMAGE_DIMENSION, IMAGE_DIMENSION))
        for object in self._geometric_objects:
            object.draw(img)
        img = img.resize((64, 64), resample=Image.ANTIALIAS)
        return img

    def get_objects(self):
        return self._geometric_objects
