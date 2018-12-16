from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.image_individual import ImageIndividual
from utils.image_utilities import ImageUtilities
from random import randint, random, sample


class BasicApproach(GeneticAlgorithm):

    def fitness(self, individual):
        individual.classify()
        return individual.classification.value_for_class(self.class_to_optimize)

    def grade(self, population):
        return max(self.fitness(x) for x in population)


