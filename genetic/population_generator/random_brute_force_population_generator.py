import string

from classifier.online_classifier import Classifier
from genetic.image_individual import ImageIndividual
from genetic.population_generator.random_population_generator import RandomPopulationGenerator


class RandomBruteForcePopulationGenerator(RandomPopulationGenerator):
    """
    Generates a population of random image individuals, while ensuring that each individual's classification contains
    the target class.
    """

    def __init__(self, size: int, classifier: Classifier, target_class: string):
        super().__init__(size=size)

        self._classifier = classifier
        self._target_class = target_class

    def __iter__(self):
        """
        This function is called when the population generator is iterated. It yields a set image individuals.
        :return: Yields image individuals
        """
        for i in range(self.size):

            current_image = self._generate_noise()
            current_classification = self._classifier.classify(current_image)

            while current_classification.value_for_class(self._target_class) <= 0:
                current_image = self._generate_noise()
                current_classification = self._classifier.classify(current_image)

            individual = ImageIndividual(image=current_image)
            individual.classification = current_classification
            yield individual
            self._progress_bar_step()

    def __repr__(self):
        return "Random Brute Force Population Generator"
