from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.image_individual import ImageIndividual
from classifier.classifier import Classifier
from typing import List
from math import inf

import numpy as np
import string


class GradeAverage(GeneticAlgorithm):
    """
    Overwrites the grade function which now returns the average fitness of the population.
    """

    __NO_AVERAGE_SIZE = -1

    def __init__(self, classifier: Classifier, class_to_optimize: string, retain_rate: float = 0.2,
                 random_select_rate: float = 0.00, mutation_rate: float = 0.02,
                 mutation_intensity = 0.05, average_size = __NO_AVERAGE_SIZE):
        super().__init__(classifier=classifier,
                         class_to_optimize=class_to_optimize,
                         retain_rate=retain_rate,
                         random_select_rate=random_select_rate,
                         mutation_rate=mutation_rate,
                         mutation_intensity=mutation_intensity)

        self._average_size = average_size


    def _grade(self, population: List[ImageIndividual] = None) -> float:
        if not population:
            population = self._get_current_population()


        fitness_list = [self._fitness(x) for x in population]

        if self._average_size != GradeAverage.__NO_AVERAGE_SIZE:
            assert self._average_size > 0
            fitness_list = sorted(fitness_list, reverse=True)[:self._average_size]

        return np.mean(fitness_list)