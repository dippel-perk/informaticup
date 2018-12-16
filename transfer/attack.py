import foolbox
import numpy as np
from foolbox.criteria import TargetClassProbability, TargetClass, ConfidentMisclassification
from keras import backend as K
from PIL import Image


class FoolAttack:
    def __init__(self, model):
        self._fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))

    def run(self, image, label):
        image = np.asarray(image, dtype=np.float32)
        image = image / 255
        adv = Image.open('test.ppm').resize((64, 64))
        adv = np.asarray(adv, dtype=np.float32) / 255

        attack = foolbox.attacks.PGD(self._fmodel, criterion=TargetClassProbability(12, p=0.99))
        adversarial = attack(image, label)
        return adversarial


class GradientAttack:
    def __init__(self, model):
        self._model = model

    def run(self, image):
        img = np.asarray(image, dtype=np.float32)
        img /= 255
        img = np.expand_dims(img, axis=0)

        original_img = img.copy()

        max_change_above = img + 0.01
        max_change_below = img - 0.01

        object_type_to_fake = 0
        model_input_layer = self._model.layers[0].input
        model_output_layer = self._model.layers[-1].output

        learning_rate = 0.1

        # Define the cost function.
        # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
        cost_function = model_output_layer[0, object_type_to_fake]

        # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
        # In this case, referring to "model_input_layer" will give us back image we are hacking.
        gradient_function = K.gradients(cost_function, model_input_layer)[0]

        # Create a Keras function that we can call to calculate the current cost and gradient
        grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                        [cost_function, gradient_function])

        cost = 0.0

        iterations = 0

        # In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
        # until it gets to at least 80% confidence
        while cost < 0.999 or object_type_to_fake > 42:
            # Check how close the image is to our target class and grab the gradients we
            # can use to push it one more step in that direction.
            # Note: It's really important to pass in '0' for the Keras learning mode here!
            # Keras layers behave differently in prediction vs. train modes!
            cost, gradients = grab_cost_and_gradients_from_model([img, 0])
            print(cost)

            # Move the hacked image one step further towards fooling the model
            img += gradients * learning_rate
            #img = np.clip(img, max_change_below, max_change_above)
            #img = np.clip(img, -1.0, 1.0)
            iterations += 1

            if iterations % 50 == 0:
                img = original_img
                object_type_to_fake += 1
                cost_function = model_output_layer[0, object_type_to_fake]

                # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
                # In this case, referring to "model_input_layer" will give us back image we are hacking.
                gradient_function = K.gradients(cost_function, model_input_layer)[0]

                # Create a Keras function that we can call to calculate the current cost and gradient
                grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                                [cost_function, gradient_function])

        return (img).reshape(64, 64, 3)