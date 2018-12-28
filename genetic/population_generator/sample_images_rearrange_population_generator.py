import os
import random as rd
from PIL import Image, ImageOps
from genetic.population_generator.population_generator import PopulationGenerator
from genetic.image_individual import ImageIndividual
from utils.image_utilities import ImageUtilities
from classifier.classifier import Classifier

class SampleImagesRearrangePopulationGenerator(PopulationGenerator):

    def __init__(self, size : int, target_class: int, image_dir: str):
        super().__init__(size)
        self._directory = os.fsencode(os.path.join(image_dir, str(target_class).zfill(5)))

    def __iter__(self):
        files = rd.sample([file for file in os.listdir(self._directory) if os.fsdecode(file).endswith(".ppm")], self.size)

        for file in files:
            image = Image.open(os.path.join(self._directory, file))
            image = ImageOps.fit(image,
                                 Classifier.DESIRED_IMAGE_DIMENSIONS,
                                 Image.ANTIALIAS)

            ImageUtilities.rearrange_image(image)
            yield ImageIndividual(image=image)