import string
from typing import List

import numpy as np

from classifier.classifier import Classifier
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.image_individual import ImageIndividual
from utils.image_utilities import ImageUtilities


class GradeAverageGeneticAlgorithm(GeneticAlgorithm):
    """
    Overwrites the grade function which now returns the average fitness of the population.
    """

    __NO_AVERAGE_SIZE = -1

    def __init__(self, classifier: Classifier, class_to_optimize: string, retain_rate: float = 0.2,
                 random_select_rate: float = 0.00, mutation_rate: float = 1.0, mutation_intensity=0.05,
                 save_generations: bool = True, output_dir: str = None,
                 pixel_mutation_function=ImageUtilities.mutate_pixels,
                 average_size=__NO_AVERAGE_SIZE):
        super().__init__(classifier=classifier,
                         class_to_optimize=class_to_optimize,
                         retain_rate=retain_rate,
                         random_select_rate=random_select_rate,
                         mutation_rate=mutation_rate,
                         mutation_intensity=mutation_intensity,
                         pixel_mutation_function=pixel_mutation_function,
                         save_generations=save_generations,
                         output_dir=output_dir)

        self._average_size = average_size

    def _grade(self, population: List[ImageIndividual] = None) -> float:
        """
        Alternative grade function. Grades a population by its average fitness instead of the maximum fitness.
        If no population is provided, the current population is used.
        :param population: The population which should be graded.
        :return: The grade.
        """
        if not population:
            population = self._get_current_population()

        fitness_list = [self._fitness(x) for x in population]

        if self._average_size != GradeAverageGeneticAlgorithm.__NO_AVERAGE_SIZE:
            assert self._average_size > 0
            fitness_list = sorted(fitness_list, reverse=True)[:self._average_size]

        return np.mean(fitness_list)

    def __repr__(self):
        return "Grade Average Genetic Algorithm"
