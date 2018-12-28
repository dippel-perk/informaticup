import string

class Classifier():

    DESIRED_IMAGE_EXTENSION = "PNG"
    DESIRED_IMAGE_WIDTH = 64
    DESIRED_IMAGE_HEIGHT = 64

    DESIRED_IMAGE_DIMENSIONS = (DESIRED_IMAGE_WIDTH, DESIRED_IMAGE_HEIGHT)

    def classify(self, file_name : string):
       raise NotImplementedError


