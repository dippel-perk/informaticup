import requests
import time


from classification import ImageClassification, Class

class NeuralNetworkAPI():

    DESIRED_IMAGE_EXTENSION = "PNG"
    DESIRED_IMAGE_WIDTH = 64
    DESIRED_IMAGE_HEIGHT = 64
    RATE_LIMIT_INTERVAL = 60

    __API_URL = 'https://phinau.de/trasi'
    __API_KEY = 'EiCheequoobi0WuPhai3saiLud4ailep'

    __API_RESPONSE_CODE_TOO_MANY_REQUESTS = 429

    def __init__(self):
        self._time_start = 0
        self._counter = 0

    def classify(self, file_name):
        if self._counter == 0:
            self._time_start = time.time()
        self._counter += 1

        data = {
            'key': NeuralNetworkAPI.__API_KEY
        }
        files = {
            'image': open(file_name, 'rb')
        }
        resp = requests.post(NeuralNetworkAPI.__API_URL, data=data, files=files)

        if resp.status_code == NeuralNetworkAPI.__API_RESPONSE_CODE_TOO_MANY_REQUESTS:
            elapsed = time.time() - self._time_start
            wait_time = NeuralNetworkAPI.RATE_LIMIT_INTERVAL - elapsed
            self._counter = 0
            print("[Warning] Too many requests, waiting {} seconds".format(wait_time))
            time.sleep(wait_time)
            return self.classify(file_name)

        data = resp.json()

        classes = list()

        for cl in data:
            classes.append(Class(cl["class"], cl["confidence"]))

        return ImageClassification(file_name, classes)


