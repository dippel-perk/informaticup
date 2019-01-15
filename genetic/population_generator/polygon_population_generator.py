import random as rd

import numpy as np

from config.geometric_individual_configuration import GeometricIndividualConfiguration
from genetic.geometric.geometric_individual import GeometricIndividual
from genetic.geometric.geometric_objects import Polygon
from genetic.population_generator.population_generator import PopulationGenerator


class PolygonPopulationGenerator(PopulationGenerator):
    """
    Generates a population of geometric individuals which contain random polygons.
    """

    def __init__(self, size: int, n=3, avg_num=200, std_num=1):
        super().__init__(size=size)
        self._n = n
        self._avg_num = avg_num
        self._std_num = std_num

    def __iter__(self):
        """
        This function is called when the population generator is iterated. It yields a set geometric individuals.
        :return: Yields geometric individuals
        """
        for i in range(self.size):
            polygons = []
            for j in range(int(np.random.normal(self._avg_num, self._std_num))):
                points = [rd.randint(0, GeometricIndividualConfiguration.IMAGE_WIDTH) for _ in range(2 * self._n)]
                color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
                polygons.append(Polygon(points=points, color=color))
            yield GeometricIndividual(polygons)
            self._progress_bar_step()

    def __repr__(self):
        return "Polygon Population Generator"
