from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.image_individual import ImageIndividual
from classifier.classifier import Classifier
from utils.image_utilities import ImageUtilities
from typing import List

import string

import random as rd

class BasicApproach(GeneticAlgorithm):

    def __init__(self, classifier: Classifier, class_to_optimize: string, retain_rate: float = 0.2,
                 random_select_rate: float = 0.00, mutation_rate: float = 0.02, mutation_intensity = 0.05):
        super().__init__(classifier=classifier,
                         class_to_optimize=class_to_optimize,
                         retain_rate=retain_rate,
                         random_select_rate=random_select_rate,
                         mutation_rate=mutation_rate,
                         mutation_intensity=mutation_intensity)

        self._base_mutation_intensity = mutation_intensity

    def _grade(self, population: List[ImageIndividual] = None):
        if not population:
            population = self._get_current_population()
        return max(self._fitness(x) for x in population)

    def _post_evolve(self) -> None:
        """
        Increases the mutation intensity iff the fitness did not improve during the evolve
        :return: None
        """
        if len(self._fitness_history) < 2:
            return
        if self._fitness_history[-1] <= self._fitness_history[-2]:
            self._mutation_intensity += self._base_mutation_intensity
        else:
            self._mutation_intensity = self._base_mutation_intensity



    #def _mutate(self, individual: ImageIndividual):
    #    assert individual.image
    #    print("Rearranging Mutation")
    #    individual.image = ImageUtilities.rearrange_image(individual.image)


