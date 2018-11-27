import requests
import json

url = 'https://phinau.de/trasi'


def classify(file_name):
    data = {
        'key': 'EiCheequoobi0WuPhai3saiLud4ailep'
    }
    files = {
        'image': open(file_name, 'rb')
    }
    resp = requests.post(url, data=data, files=files)
    return resp.json()
