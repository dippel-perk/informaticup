from transfer.classifier import OfflineClassifier
import pandas as pd
from transfer.attack import FoolAttack
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from PIL import Image

    img = Image.open('cat.jpg')
    noise = np.random.random((64, 64, 3)) * 255
    img = Image.fromarray(noise.astype('uint8'), 'RGB')
    new_img = img.resize((64, 64))
    img_path = "tmp/image.png"
    new_img.save(img_path, "PNG", optimize=True)


    #result = classify(img_path)
    #print(result)

    prefix = '../GTSRB/'
    test_df = pd.read_csv(prefix + 'Final_Test/test.csv', delimiter=';')
    classifier = OfflineClassifier(weights_file='weights-08-0.95.hdf5')

    attack = FoolAttack(model=classifier.get_model())
    class_id = classifier.predict(img_path)

    result = attack.run(new_img, class_id)

    plt.imshow(result)
    plt.show()

    img_adv = Image.fromarray((result * 255).astype('uint8'), 'RGB')
    img_adv.save(img_path, "PNG", optimize=True)
    #print(classify(img_path))
    #classifier.predict(img_path)


    #classifier.predict(img_path)

