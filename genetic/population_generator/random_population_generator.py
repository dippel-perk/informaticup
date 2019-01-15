import random as rd

from PIL import Image

from genetic.image_individual import ImageIndividual
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities


class RandomPopulationGenerator(PopulationGenerator):
    """
    Generates a population of random image individuals.
    """

    def __init__(self, size: int):
        super().__init__(size=size)

    def _generate_noise(self) -> Image:
        """
        Generates a random noise image. To get a random pixel value the get pixel value function is used.
        :return: The random noise image.
        """
        img, pixel_count = ImageUtilities.get_empty_image()

        pixel_data = list()

        for i in range(pixel_count):
            pixel_data.append(self._get_pixel_value())

        img.putdata(pixel_data)

        return img

    def _get_pixel_value(self) -> tuple:
        """
        Generates random pixel value.
        :return: The pixel value.
        """
        pixel_val = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
        return pixel_val

    def __iter__(self):
        """
        This function is called when the population generator is iterated. It yields a set image individuals.
        :return: Yields image individuals
        """
        for i in range(self.size):
            yield ImageIndividual(image=self._generate_noise())
            self._progress_bar_step()

    def __repr__(self):
        return "Random Population Generator"
