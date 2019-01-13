import numpy as np
import random as rd

from genetic.population_generator.population_generator import PopulationGenerator
from genetic.geometric.geometric_objects import Bitmap
from genetic.geometric.geometric_individual import GeometricIndividual
from config.geometric_individual_configuration import GeometricIndividualConfiguration

class BitmapPopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, img, num_horizontal=4, num_vertical=4):
        super().__init__(size=size)
        self._img = img
        self._num_horizontal = num_horizontal
        self._num_vertical = num_vertical

    def __iter__(self):
        for i in range(self.size):
            bitmaps = []

            dimensions = GeometricIndividualConfiguration.IMAGE_DIMENSION
            width = dimensions[0]
            height = dimensions[1]

            img  = self._img.resize((int(width / self._num_horizontal), int(height / self._num_vertical)))

            for x in range(0, width, int(width / self._num_horizontal)):
                for y in range(0, height, int(height / self._num_vertical)):
                    color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
                    bitmaps.append(Bitmap(img=img, x=x, y=y, color=color))
            yield GeometricIndividual(bitmaps)

    def __repr__(self):
        return "Bitmap Population Generator"
