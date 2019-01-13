import numpy as np

from genetic.population_generator.population_generator import PopulationGenerator
from genetic.geometric.geometric_objects import Circle
from genetic.geometric.geometric_individual import GeometricIndividual

class CirclePopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, avg_num_circles=200, std_num_circles=1, avg_radius_size=70, std_radius_size=30):
        super().__init__(size=size)
        self._avg_num_circles = avg_num_circles
        self._std_num_circles = std_num_circles
        self._avg_radius_size = avg_radius_size
        self._std_radius_size = std_radius_size

    def _generate_circles(self):
        circles = []
        num_circles = int(np.random.normal(self._avg_num_circles, self._std_num_circles))
        for i in range(num_circles):
            circles.append(Circle.generate(self._avg_radius_size, self._std_radius_size))
        return circles

    def __iter__(self):
        for i in range(self.size):
            yield GeometricIndividual(self._generate_circles())

    def __repr__(self):
        return "Circle Population Generator"
