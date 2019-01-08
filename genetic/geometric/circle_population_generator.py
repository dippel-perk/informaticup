from genetic.population_generator.population_generator import PopulationGenerator
from genetic.geometric.geometric_objects import Circle
from genetic.geometric.geometric_individual import GeometricIndividual
import numpy as np
import random as rd


def generate_circle(avg_radius, std_radius):
    x = np.random.randint(0, 63)
    y = np.random.randint(0, 63)
    radius = np.random.normal(avg_radius, std_radius)
    color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
    return Circle(x=x, y=y, radius=radius, color=color)


class CirclePopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, avg_num_circles=60, std_num_circles=1, avg_radius_size=3, std_radius_size=3):
        super().__init__(size=size)
        self._avg_num_circles = avg_num_circles
        self._std_num_circles = std_num_circles
        self._avg_radius_size = avg_radius_size
        self._std_radius_size = std_radius_size

    def _generate_circles(self):
        circles = []
        num_circles = int(np.random.normal(self._avg_num_circles, self._std_num_circles))
        for i in range(num_circles):
            circles.append(generate_circle(self._avg_radius_size, self._std_radius_size))
        return circles

    def __iter__(self):
        for i in range(self.size):
            yield GeometricIndividual(self._generate_circles())
