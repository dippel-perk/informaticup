from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.image_individual import ImageIndividual
from typing import List

class BasicApproach(GeneticAlgorithm):

    def _grade(self, population: List[ImageIndividual] = None):
        if not population:
            population = self._get_current_population()
        return max(self._fitness(x) for x in population)


