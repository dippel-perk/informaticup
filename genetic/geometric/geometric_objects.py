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


class Polygon:
    def __init__(self, points, color):
        self.points = points
        self.color = color

    def draw(self, image: Image):
        draw = ImageDraw.Draw(image)
        draw.polygon(self.points, self.color)


class Bitmap:
    def __init__(self, x, y, img, color):
        self.img = img
        self.x = x
        self.y = y
        self.color = color

    def draw(self, image: Image):
        draw = ImageDraw.Draw(image)
        draw.bitmap((self.x, self.y), bitmap=self.img, fill=self.color)

