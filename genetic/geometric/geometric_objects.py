import random as rd
import numpy as np

from config.geometric_individual_configuration import GeometricIndividualConfiguration
from PIL import Image, ImageDraw

class GeometricObject:
    """
    Base class for geometric objects.
    """
    def draw(self, image: Image) -> None:
        """
        This method should be implemented by the subclasses. It should generate the current geometric object
        into the given image.
        :param image: The image into which the object should be generated.
        :return: None
        """
        raise NotImplementedError

class Circle(GeometricObject):
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def draw(self, image: Image) -> None:
        """
        Generates the circle into the given image.
        :param image: The input image.
        :return: None
        """
        draw = ImageDraw.Draw(image)
        draw.ellipse((self.x - self.radius, self.y - self.radius, self.x + self.radius, self.y + self.radius),
                     fill=self.color)

    @staticmethod
    def generate(avg_radius: int, std_radius: int):
        #TODO: Return Circle geht nicht?
        """
        Generates a circle based on an expected radius and a standard distribution
        at a random position with a random color.
        :param avg_radius: The expected radius.
        :param std_radius: The standard distribution.
        :return: The generated Circle
        """
        x = np.random.randint(0, GeometricIndividualConfiguration.IMAGE_WIDTH)
        y = np.random.randint(0, GeometricIndividualConfiguration.IMAGE_WIDTH)
        radius = max(np.random.normal(avg_radius, std_radius), 0)
        color = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
        return Circle(x=x, y=y, radius=radius, color=color)
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.radius == other.radius and self.color == other.color


class Polygon(GeometricObject):
    def __init__(self, points, color):
        self.points = points
        self.color = color

    def draw(self, image: Image) -> None:
        """
        Generates the polygon into the given image.
        :param image: The input image.
        :return: None
        """
        draw = ImageDraw.Draw(image)
        draw.polygon(self.points, self.color)


class Bitmap(GeometricObject):
    def __init__(self, x, y, img, color):
        self.img = img
        self.x = x
        self.y = y
        self.color = color

    def draw(self, image: Image) -> None:
        """
        Generates the bitmap into the given image.
        :param image: The input image.
        :return: None
        """
        draw = ImageDraw.Draw(image)
        draw.bitmap((self.x, self.y), bitmap=self.img, fill=self.color)


class Tile(GeometricObject):

    def __init__(self, coordinates, color):
        self.coordinates = coordinates
        self.color = color

    def draw(self, image: Image) -> None:
        """
        Draws the tile into the image.
        :param image: The input image.
        :return: None
        """
        draw = ImageDraw.Draw(image)
        draw.polygon(self.coordinates, fill=self.color)


