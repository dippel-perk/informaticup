from classifier.online_classifier import OnlineClassifier
from PIL import Image, ImageOps
import pandas as pd
from utils.image_utilities import ImageUtilities
import os, random
from road_sign_class_mapper import RoadSignClassMapper


image_path = '../GTSRB/Final_Training/Images'
data = []
classifier = OnlineClassifier()
df = None
mapper = RoadSignClassMapper()
for i in range(43):
    if not mapper.get_name_by_class(i) is None:
        continue

    dir_name = os.path.join(image_path, str(i).zfill(5))
    for k in range(20):
        file = random.choice(os.listdir(dir_name))
        image = Image.open(os.path.join(dir_name, file))
        image = ImageOps.fit(image,
                             OnlineClassifier.DESIRED_IMAGE_DIMENSIONS,
                             Image.ANTIALIAS)
        file = ImageUtilities.save_image_to_tempfile(image)
        classification = classifier.classify(file)
        entry = {'class_id': i}
        c = sorted(classification.classes, key=lambda x: x.confidence, reverse=True)[0]
        entry['class'] = c.name
        entry['confidence'] = c.confidence
        data.append(entry)
    df = pd.DataFrame(data)
    print(df.tail(20))

df.to_csv('class_names.csv')
