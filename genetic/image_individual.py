from PIL import Image


class ImageIndividual:
    def __init__(self, image: Image):
        self.image = image
        self.classification = None

    def __len__(self):
        width, height = self.image.size
        return width * height

    def __repr__(self):
        return self.classification.__repr__()
