import time
import requests
import os

from PIL.Image import Image
from json import JSONDecodeError
from typing import List
from utils.output_utilities import print_error, print_debug,print_countdown
from classifier.classifier import Classifier
from classifier.classification import ImageClassification, Class
from utils.image_utilities import ImageUtilities
from config.classifier_configuration import OnlineClassifierConfiguration


class OnlineClassifier(Classifier):

    __API_RESPONSE_CODE_TOO_MANY_REQUESTS = 429
    __API_RESPONSE_CODE_BAD_REQUEST = 400
    __API_RESPONSE_CODE_SERVICE_UNAVAILABLE = 503
    __API_RESPONSE_CODE_UNAUTHORIZED = 401

    __API_RATE_LIMIT_INTERVAL = 60


    def __init__(self):
        self._time_start = 0
        self._counter = 0

    def classify(self, image: Image) -> ImageClassification:
        """
        Classifies a single image with the online network and returns the classification.
        I case of an error an error message gets printed to the terminal and an exception might be thrown.
        :param image: The input image.
        :return: The classification result.
        """

        if self._counter == 0:
            self._time_start = time.time()
        self._counter += 1

        file = ImageUtilities.save_image_to_tempfile(image)

        data = {'key': OnlineClassifierConfiguration.API_KEY}
        files = {'image': open(file, 'rb')}

        resp = requests.post(OnlineClassifierConfiguration.API_URL, data=data, files=files)

        os.remove(file)

        if resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_BAD_REQUEST:
            print_error("Online Classifier: Bad Request")
            print_debug(resp)
            raise ConnectionRefusedError("Bad Request")

        elif resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_UNAUTHORIZED:
            print_error("Online Classifier: Unauthorized Request")
            print_debug("Used API key: " + OnlineClassifierConfiguration.API_KEY)
            raise ConnectionRefusedError("Unauthorized Request")

        elif resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_SERVICE_UNAVAILABLE:
            print_error("Online Classifier: Service Unavailable")
            print_debug(resp)
            print_countdown(wait_time=10, prefix_text="Trying again in")
            return self.classify(image)

        elif resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_TOO_MANY_REQUESTS:
            elapsed = time.time() - self._time_start
            wait_time = int(max(OnlineClassifier.__API_RATE_LIMIT_INTERVAL - elapsed, 0)) + 1
            self._counter = 0
            print_countdown(wait_time=wait_time, prefix_text="Too many requests. Trying again in")
            return self.classify(image)

        try:
            data = resp.json()
        except JSONDecodeError:
            print_error("Online Classifier: Could not decode JSON")
            print_debug(resp)
            return self.classify(file)

        classes = list()

        for cl in data:
            classes.append(Class(cl["class"], cl["confidence"]))

        return ImageClassification(classes)

    def classify_batch(self, images: List[Image]) -> List[ImageClassification]:
        """
        Classifies a list of images with the online network and returns the classifications.
        :param images: The input images.
        :return: The list of classification results.
        """
        return [self.classify(image) for image in images]

    def __repr__(self):
        return 'OnlineClassifier'
