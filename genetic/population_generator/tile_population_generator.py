import random as rd
import math

from genetic.population_generator.population_generator import PopulationGenerator
from genetic.geometric.geometric_individual import GeometricIndividual
from genetic.geometric.geometric_objects import Tile
from config.geometric_individual_configuration import GeometricIndividualConfiguration


def interpolation_color_generator(color1, color2):
    """
    Generate random colors between color1 and color2.
    """
    # Get the difference along each axis
    d_red = color1[0] - color2[0]
    d_green = color1[1] - color2[1]
    d_blue = color1[2] - color2[2]

    while True:
        proportion = rd.uniform(0, 1)

        yield (color1[0] - int(d_red * proportion),
               color1[1] - int(d_green * proportion),
               color1[2] - int(d_blue * proportion))


def random_color_generator():
    while True:
        yield (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))


class TilePopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, color1=(255, 82, 82), color2=(255, 255, 255), scale_factor=50, interpolate=True):
        super().__init__(size=size)
        self._color1 = color1
        self._color2 = color2
        self._interpolate = interpolate
        self._scale_factor = scale_factor

    def generate_unit_triangles(self, image_width, image_height):
        """
        Generate coordinates for a tiling of unit triangles.
        :param image_width: The image width.
        :param image_height: The image height.
        :return: Yields all triangles of the image.
        """
        h = math.sin(math.pi / 3)

        # The first triangle starts beyond the left-hand side of the image,
        # and is only partially visible.  This lets us cover the whole image.
        # Likewise we add an extra row to cover the bottom.
        for x in range(-1, image_width):
            for y in range(int(image_height / h) + 1):
                # Add a horizontal offset on odd numbered rows
                x_ = x if (y % 2 == 0) else x + 0.5

                yield [(x_, y * h), (x_ + 1, y * h), (x_ + 0.5, (y + 1) * h)]
                yield [(x_ + 1, y * h), (x_ + 1.5, (y + 1) * h), (x_ + 0.5, (y + 1) * h)]

    def _scale_coordinates(self, generator, image_width, image_height, side_length=1):
        scaled_width = int(image_width / side_length) + 2
        scaled_height = int(image_height / side_length) + 2

        for coords in generator(scaled_width, scaled_height):
            yield [(x * side_length, y * side_length) for (x, y) in coords]

    def __iter__(self):
        for i in range(self.size):
            shapes = self._scale_coordinates(self.generate_unit_triangles,
                                             GeometricIndividualConfiguration.IMAGE_DIMENSION[0],
                                             GeometricIndividualConfiguration.IMAGE_DIMENSION[1],
                                             100)
            colors = None
            if self._interpolate:
                colors = interpolation_color_generator(self._color1, self._color2)
            else:
                colors = random_color_generator()

            tiles = [Tile(shape, color) for shape, color in zip(shapes, colors)]
            yield GeometricIndividual(tiles)

    def __repr__(self):
        return "Tile Population Generator"