import foolbox
from foolbox.criteria import TargetClassProbability, TargetClass, ConfidentMisclassification
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class OnlineModel(foolbox.models.Model):

    def batch_predictions(self, images):
        img_path = "tmp/test.png"
        for img in images:
            plt.imshow(img)
            plt.show()
            image = Image.fromarray((img * 255).astype('uint8'), 'RGB')
            image.save(img_path)
            pass

    def num_classes(self):
        return 43



if __name__ == '__main__':
    img = Image.open('cat.jpg')
    noise = np.random.random((64, 64, 3)) * 255
    # img = Image.fromarray(noise.astype('uint8'), 'RGB')
    new_img = img.resize((64, 64))
    img_path = "tmp/image.png"
    new_img.save(img_path, "PNG", optimize=True)
    model = OnlineModel(bounds=(0, 1), channel_axis=0)

    image = np.asarray(new_img, dtype=np.float32)
    image = image / 255

    adv = Image.open('test.ppm').resize((64, 64))
    adv = np.asarray(adv, dtype=np.float32) / 255

    attack = foolbox.attacks.PointwiseAttack(model, criterion=TargetClassProbability(12, p=0.9))
    adversarial = attack(image, 0, starting_point=adv)
