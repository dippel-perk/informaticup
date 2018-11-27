from api import classify
from classifier import OfflineClassifier
import pandas as pd

if __name__ == '__main__':
    from PIL import Image

    img = Image.open('test.ppm')
    new_img = img.resize((64, 64))
    img_path = "tmp/image.png"
    new_img.save(img_path, "PNG", optimize=True)

    prefix = '../GTSRB/'
    test_df = pd.read_csv(prefix + 'Final_Test/test.csv', delimiter=';')[:1000]
    classifier = OfflineClassifier()
    classifier.train(train_path=prefix + 'Final_Training/Images',
                     test_image_path=prefix + 'Final_Test/Images',
                     test_df=test_df)
