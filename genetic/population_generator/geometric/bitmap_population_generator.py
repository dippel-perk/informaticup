import numpy as np
import random as rd

from genetic.population_generator.population_generator import PopulationGenerator
from genetic.geometric.geometric_objects import Bitmap
from genetic.geometric.geometric_individual import GeometricIndividual

class BitmapPopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, img, avg_num=20, std_num=5):
        super().__init__(size=size)
        self._img = img
        self._avg_num = avg_num
        self._std_num = std_num

    def __iter__(self):
        for i in range(self.size):
            bitmaps = []
            for j in range(int(np.random.normal(self._avg_num, self._std_num))):
                x, y = [rd.randint(0, GeometricIndividual.IMAGE_DIMENSION) for _ in range(2)]
                color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
                bitmaps.append(Bitmap(img=self._img, x=x, y=y, color=color))
            yield GeometricIndividual(bitmaps)
