from typing import List

import numpy as np

from genetic.geometric.geometric_individual import GeometricIndividual
from genetic.geometric.geometric_objects import Circle
from genetic.population_generator.population_generator import PopulationGenerator


class CirclePopulationGenerator(PopulationGenerator):
    """
    Generates a population of geometric individuals, which are filled with random circles.
    """

    def __init__(self, size: int, avg_num_circles=200, std_num_circles=1, avg_radius_size=70, std_radius_size=30):
        super().__init__(size=size)
        self._avg_num_circles = avg_num_circles
        self._std_num_circles = std_num_circles
        self._avg_radius_size = avg_radius_size
        self._std_radius_size = std_radius_size

    def _generate_circles(self) -> List[Circle]:
        """
        Generates a list of circles according to the configuration of the population generator object.
        :return: List of circles.
        """
        circles = []
        num_circles = int(np.random.normal(self._avg_num_circles, self._std_num_circles))
        for i in range(num_circles):
            circles.append(Circle.generate(self._avg_radius_size, self._std_radius_size))
        return circles

    def __iter__(self):
        """
        This function is called when the population generator is iterated. It yields a set image individuals.
        :return: Yields image individuals
        """
        for i in range(self.size):
            yield GeometricIndividual(self._generate_circles())
            self._progress_bar_step()

    def __repr__(self):
        return "Circle Population Generator"
