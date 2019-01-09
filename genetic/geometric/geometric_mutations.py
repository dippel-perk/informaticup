from genetic.geometric.geometric_objects import GeometricObject
from genetic.geometric.circle_population_generator import generate_circle
import random as rd
from genetic.geometric.geometric_individual import IMAGE_DIMENSION
from genetic.geometric.geometric_objects import Polygon, Bitmap
from PIL.Image import Image


class GeometricMutations:

    @staticmethod
    def mutate_circle(object: GeometricObject):
        return generate_circle(10, 5)

    @staticmethod
    def mutate_polygon_function(dimension: int):
        def mutate_polygon(object: GeometricObject):
            points = [rd.randint(0, IMAGE_DIMENSION) for _ in range(2 * dimension)]
            color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
            return Polygon(points=points, color=color)

        return mutate_polygon

    @staticmethod
    def mutate_bitmap_function(img: Image):
        def mutate_bitmap(object: GeometricObject):
            x, y = [rd.randint(0, IMAGE_DIMENSION) for _ in range(2)]
            color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
            bitmap = Bitmap(img=img, x=x, y=y, color=color)
            return bitmap
        return mutate_bitmap
