from genetic.population_generator.population_generator import PopulationGenerator
from genetic.image_individual import ImageIndividual
import foolbox
from classifier.neural_net import NeuralNet
from foolbox.criteria import TargetClassProbability
import numpy as np
from PIL import Image
from keras import backend as K


class GradientPopulationGenerator(PopulationGenerator):

    def __init__(self, size, class_id: int, population_generator: PopulationGenerator):
        super().__init__(size=size)
        self._class_id = class_id
        self._population_generator = population_generator
        self._neural_net = NeuralNet()
        self._model = self._neural_net.get_model()
        self._fmodel = foolbox.models.KerasModel(self._model, bounds=(0, 1))

    def __iter__(self):
        for individual in self._population_generator:
            yield ImageIndividual(image=self._gradient_attack2(individual.image))

    def _gradient_attack(self, image):
        image = np.asarray(image, dtype=np.float32)
        image = image / 255
        attack = foolbox.attacks.LBFGSAttack(self._fmodel, criterion=TargetClassProbability(self._class_id, p=0.2))
        adversarial = attack(image, 0)
        return Image.fromarray((adversarial * 255).astype('uint8'), 'RGB')

    def _gradient_attack2(self, image):
        img = np.asarray(image, dtype=np.float32)
        img /= 255
        img = np.expand_dims(img, axis=0)

        object_type_to_fake = self._class_id
        model_input_layer = self._model.layers[0].input
        model_output_layer = self._model.layers[-1].output

        learning_rate = 3

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
        while cost < 0.9:
            # Check how close the image is to our target class and grab the gradients we
            # can use to push it one more step in that direction.
            # Note: It's really important to pass in '0' for the Keras learning mode here!
            # Keras layers behave differently in prediction vs. train modes!
            cost, gradients = grab_cost_and_gradients_from_model([img, 0])
            print(cost)

            # Move the hacked image one step further towards fooling the model
            img += gradients * learning_rate
            iterations += 1

        return (img).reshape(64, 64, 3)
