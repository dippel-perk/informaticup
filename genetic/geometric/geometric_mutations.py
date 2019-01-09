import random as rd

from PIL.Image import Image
from config.geometric_individual_configuration import GeometricIndividualConfiguration
from genetic.geometric.geometric_objects import GeometricObject
from genetic.geometric.geometric_objects import Polygon, Bitmap
from genetic.geometric.geometric_objects import Circle

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
    def mutate_bitmap_function(img: Image):
        """
        Generates a mutation function which generates the given image with a random color
        at a random position.
        :param img: The input image.
        :return: The mutation function.
        """
        def mutate_bitmap(object: GeometricObject):
            x, y = [rd.randint(0, GeometricIndividualConfiguration.IMAGE_WIDTH) for _ in range(2)]
            color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
            bitmap = Bitmap(img=img, x=x, y=y, color=color)
            return bitmap
        return mutate_bitmap
