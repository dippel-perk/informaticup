import pandas as pd
from transfer.classifier import OfflineClassifier


prefix = '../GTSRB/'
test_df = pd.read_csv(prefix + 'Final_Test/test.csv', delimiter=';')
classifier = OfflineClassifier()
classifier.train(train_path=prefix + 'Final_Training/Images',
                 test_image_path=prefix + 'Final_Test/Images',
                 test_df=test_df)