import random as rd

from PIL import Image

from config.classifier_configuration import ClassifierConfiguration
from genetic.image_individual import ImageIndividual
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities


class SingleImagePopulationGenerator(PopulationGenerator):
    """
    Generates a population with geometric individuals, containing a single image with random pixel values in their center.
    """

    def __init__(self, size: int, img):
        super().__init__(size=size)
        img = img.resize(ClassifierConfiguration.DESIRED_IMAGE_DIMENSIONS, resample=Image.ANTIALIAS)
        self._img = img

    def __iter__(self):
        """
        This function is called when the population generator is iterated. It yields a set geometric individuals.
        :return: Yields geometric individuals
        """
        for i in range(self.size):
            data = self._img.getdata()
            min = 0
            max = 255
            data = [
                (rd.randint(min, max), rd.randint(min, max), rd.randint(min, max)) if pixel != (0, 0, 0) else (0, 0, 0)
                for pixel in data]
            img, pixel = ImageUtilities.get_empty_image()
            img.putdata(data)
            yield ImageIndividual(image=img)
            self._progress_bar_step()

    def __repr__(self):
        return "Single Image Population Generator"
