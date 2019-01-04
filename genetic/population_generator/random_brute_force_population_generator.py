from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.image_individual import ImageIndividual
from classifier.online_classifier import Classifier

import string

class RandomBruteForcePopulationGenerator(RandomPopulationGenerator):

    def __init__(self, size : int, classifier: Classifier, target_class: string):
        super().__init__(size=size)

        self._classifier = classifier
        self._target_class = target_class

    def __iter__(self):
        for i in range(self.size):

            current_image = self._generate_noise()
            current_classification = self._classifier.classify(current_image)

            while current_classification.value_for_class(self._target_class) <= 0:
                current_image = self._generate_noise()
                current_classification = self._classifier.classify(current_image)

            individual = ImageIndividual(image=current_image)
            individual.classification = current_classification
            yield individual