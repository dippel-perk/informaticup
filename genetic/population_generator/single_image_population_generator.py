from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities
from genetic.image_individual import ImageIndividual
import random as rd
from config.classifier_configuration import ClassifierConfiguration
from PIL import Image
from utils.image_utilities import ImageUtilities


class SingleImagePopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, img):
        super().__init__(size=size)
        img = img.resize(ClassifierConfiguration.DESIRED_IMAGE_DIMENSIONS, resample=Image.ANTIALIAS)
        self._img = img

    def __iter__(self):
        for i in range(self.size):
            data = self._img.getdata()
            min = 0
            max = 255
            data = [(rd.randint(min,max), rd.randint(min,max), rd.randint(min,max)) if pixel != (0, 0, 0) else (0, 0, 0) for pixel in data]
            img, pixel = ImageUtilities.get_empty_image()
            img.putdata(data)
            yield ImageIndividual(image=img)
