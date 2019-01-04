import time
import requests

from json import JSONDecodeError

from classifier.classifier import Classifier
from classifier.classification import ImageClassification, Class


class OnlineClassifier(Classifier):
    SEEN_CLASSES = []

    RATE_LIMIT_INTERVAL = 60

    __API_URL = 'https://phinau.de/trasi'
    __API_KEY = 'EiCheequoobi0WuPhai3saiLud4ailep'

    __API_RESPONSE_CODE_TOO_MANY_REQUESTS = 429
    __API_RESPONSE_CODE_BAD_REQUEST = 400
    __API_RESPONSE_CODE_SERVICE_UNAVAILABLE = 503
    __API_RESPONSE_CODE_UNAUTHORIZED = 401

    def __init__(self):
        self._time_start = 0
        self._counter = 0

    def classify(self, file_name):
        if self._counter == 0:
            self._time_start = time.time()
        self._counter += 1

        data = {
            'key': OnlineClassifier.__API_KEY
        }
        files = {
            'image': open(file_name, 'rb')
        }
        resp = requests.post(OnlineClassifier.__API_URL, data=data, files=files)

        if resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_BAD_REQUEST or resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_UNAUTHORIZED:
            print(resp)
            raise ValueError("WRONG API KEY OR BAD REQUEST")

        if resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_SERVICE_UNAVAILABLE:
            print(resp)
            time.sleep(10)
            return self.classify(file_name)

        if resp.status_code == OnlineClassifier.__API_RESPONSE_CODE_TOO_MANY_REQUESTS:
            elapsed = time.time() - self._time_start
            wait_time = max(OnlineClassifier.RATE_LIMIT_INTERVAL - elapsed, 0)
            self._counter = 0
            print("[Warning] Too many requests, waiting {} seconds".format(wait_time))
            time.sleep(wait_time)
            return self.classify(file_name)

        try:
            data = resp.json()
        except JSONDecodeError:
            print("Could not decode JSON:", resp)
            return self.classify(file_name)

        classes = list()

        for cl in data:
            if cl["class"] not in OnlineClassifier.SEEN_CLASSES:
                OnlineClassifier.SEEN_CLASSES.append(cl["class"])

            classes.append(Class(cl["class"], cl["confidence"]))

        return ImageClassification(file_name, classes)
