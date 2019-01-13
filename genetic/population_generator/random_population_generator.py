from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities
from genetic.image_individual import ImageIndividual
import random as rd

class RandomPopulationGenerator(PopulationGenerator):

    def __init__(self, size : int):
        super().__init__(size=size)

    def _generate_noise(self):

        img, pixel_count = ImageUtilities.get_empty_image()

        pixel_data = list()

        for i in range(pixel_count):
            pixel_data.append(self._get_pixel_value())

        img.putdata(pixel_data)

        return img

    def _get_pixel_value(self):
        pixel_val = (rd.randint(0,255),rd.randint(0,255), rd.randint(0,255))
        return pixel_val

    def __iter__(self):
        for i in range(self.size):
            yield ImageIndividual(image=self._generate_noise())

    def __repr__(self):
        return "Random Population Generator"