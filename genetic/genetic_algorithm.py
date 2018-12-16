from api import NeuralNetworkAPI
from genetic import basic_approach


class GeneticAlgorithm:

    def __init__(self, class_to_optimize):
        self.api = NeuralNetworkAPI()
        self.class_to_optimize = class_to_optimize

    def individual(self):
        pass

    def population(self, count):
        return [self.individual() for i in range(count)]

    def fitness(self, individual):
        pass

    def grade(self, population):
        summed = sum(self.fitness(x) for x in population)
        return summed / len(population)

    def evolve(self, population, retain=0.2, random_select=0.05, mutate=0.01):
        pass
