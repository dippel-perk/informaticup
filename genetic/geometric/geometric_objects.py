from PIL import Image, ImageDraw


class GeometricObject:
    def draw(self, image: Image):
        raise NotImplementedError


class Circle:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def draw(self, image: Image):
        draw = ImageDraw.Draw(image)
        draw.ellipse((self.x - self.radius, self.y - self.radius, self.x + self.radius, self.y + self.radius),
                     fill=self.color)
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.radius == other.radius and self.color == other.color
