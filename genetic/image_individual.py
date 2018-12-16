from classification import ImageClassification
from PIL import Image
from api import NeuralNetworkAPI
from image_utilities import ImageUtilities


class ImageIndividual:
    def __init__(self, api: NeuralNetworkAPI, image: Image, classification: ImageClassification = None):
        self.api = api
        self.image = image
        self.classification = classification

    def classify(self, force_recomputation=False):
        if not self.classification or force_recomputation:
            file = ImageUtilities.save_image_to_tempfile(self.image)
            self.classification = self.api.classify(file)

    def mutate(self, position_to_mutate):
        ImageUtilities.mutate_pixel(self.image, position_to_mutate)

    def combine(self, individual):
        image = ImageUtilities.combine_images(self.image, individual.image)
        return ImageIndividual(api=self.api, image=image)

    def combinable(self, individual):
        self.classify()
        individual.classify()

        return self.classification.share_classes(individual.classification)

    def __len__(self):
        width, height = self.image.size
        return width * height

    def __repr__(self):
        return self.classification.__repr__()
