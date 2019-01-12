import random as rd

from PIL.Image import Image
from config.geometric_individual_configuration import GeometricIndividualConfiguration
from genetic.geometric.geometric_objects import GeometricObject
from genetic.geometric.geometric_objects import Polygon, Bitmap
from genetic.geometric.geometric_objects import Circle
from genetic.geometric.geometric_objects import Tile
from genetic.population_generator.geometric.tile_population_generator import interpolation_color_generator


class GeometricMutations:
    """
    Contains various mutation function generation methods for different geometric objects. The functions are used
    during the mutation process of a geometric genetic algorithm.
    """

    @staticmethod
    def mutate_circle_function(avg_radius: int = 10, std_radius: int = 5):
        """
        Generates a circle mutation function which generates a circle based on an expected radius
        and a standard distribution at a random position with a random color.
        :param avg_radius: The expected radius.
        :param std_radius: The standard distribution.
        :return: The mutation function.
        """

        def mutate_polygon(object: GeometricObject):
            return Circle.generate(avg_radius, std_radius)

        return mutate_polygon

    @staticmethod
    def mutate_polygon_function(n: int = 3):
        """
        Generates a mutation function which generates a random polygon with the given size
        at a random position.
        :param n: The number of points the resulting polygon should have.
        :return: The mutation function.
        """

        def mutate_polygon(object: GeometricObject):
            points = [rd.randint(0, GeometricIndividualConfiguration.IMAGE_WIDTH) for _ in range(2 * n)]
            color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
            return Polygon(points=points, color=color)

        return mutate_polygon

    @staticmethod
    def mutate_bitmap_function(img: Image, num_vertical=4, num_horizontal=4):
        """
        Generates a mutation function which generates the given image with a random color at the same position.
        :param img: The input image.
        :return: The mutation function.
        """

        dimensions = GeometricIndividualConfiguration.IMAGE_DIMENSION
        width = dimensions[0]
        height = dimensions[1]

        img = img.resize((int(width / num_horizontal), int(height / num_vertical)))

        def mutate_bitmap(object: GeometricObject):
            color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
            bitmap = Bitmap(img=img, x=object.x, y=object.y, color=color)
            return bitmap

        return mutate_bitmap

    @staticmethod
    def mutate_tile_function(color1, color2):
        """
        Generates a mutation function which assigns a random color to the given tile.
        :param img: The input image.
        :return: The mutation function.
        """

        def mutate_tile(object: GeometricObject):
            color_generator = interpolation_color_generator(color1, color2)
            tile = Tile(coordinates=object.coordinates, color=next(color_generator))
            return tile

        return mutate_tile
