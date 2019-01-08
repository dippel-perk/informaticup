from genetic.population_generator.population_generator import PopulationGenerator
from genetic.geometric.geometric_objects import Polygon
from genetic.geometric.geometric_individual import GeometricIndividual
import numpy as np
import random as rd
from genetic.geometric.geometric_individual import IMAGE_DIMENSION


class PolygonPopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, dimension=3, avg_num=200, std_num=1):
        super().__init__(size=size)
        self._dimension = dimension
        self._avg_num = avg_num
        self._std_num = std_num

    def __iter__(self):
        for i in range(self.size):
            polygons = []
            for j in range(int(np.random.normal(self._avg_num, self._std_num))):
                points = [rd.randint(0, IMAGE_DIMENSION) for _ in range(2 * self._dimension)]
                color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
                polygons.append(Polygon(points=points, color=color))
            yield GeometricIndividual(polygons)
