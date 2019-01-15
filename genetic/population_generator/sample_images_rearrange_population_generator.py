import os
import random as rd

from PIL import Image, ImageOps

from config.classifier_configuration import ClassifierConfiguration
from genetic.image_individual import ImageIndividual
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities


class SampleImagesRearrangePopulationGenerator(PopulationGenerator):
    """
    Generates a population, which contains rearrangements of training set images.
    """

    def __init__(self, size: int, target_class: int, image_dir: str):
        super().__init__(size)
        self._directory = os.fsencode(os.path.join(image_dir, str(target_class).zfill(5)))

    def __iter__(self):
        """
        This function is called when the population generator is iterated. It yields a set image individuals.
        :return: Yields image individuals
        """
        files = rd.sample([file for file in os.listdir(self._directory) if os.fsdecode(file).endswith(".ppm")],
                          self.size)

        for file in files:
            image = Image.open(os.path.join(self._directory, file))
            image = ImageOps.fit(image,
                                 ClassifierConfiguration.DESIRED_IMAGE_DIMENSIONS,
                                 Image.ANTIALIAS)

            ImageUtilities.rearrange_image(image)
            yield ImageIndividual(image=image)
            self._progress_bar_step()

    def __repr__(self):
        return "Sample Images Rearrange Population Generator"
