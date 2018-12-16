from api import NeuralNetworkAPI
from genetic.image_individual import ImageIndividual
from typing import List
from genetic.population_generator.population_generator import PopulationGenerator

import random as rd

class GeneticAlgorithm:

    def __init__(self, class_to_optimize):
        self.api = NeuralNetworkAPI()
        self.class_to_optimize = class_to_optimize

    def fitness(self, individual : ImageIndividual):
        pass

    def grade(self, population : List[ImageIndividual]):
        summed = sum(self.fitness(x) for x in population)
        return summed / len(population)

    def evolve(self, population : List[ImageIndividual], retain : float = 0.2, random_select : float = 0.05, mutate : float = 0.01):
        graded = sorted(population, key=lambda x: self.fitness(x), reverse=True)

        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]

        # randomly add other individuals to
        # promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > rd.random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if mutate > rd.random():
                self.mutate(individual=individual)

        # crossover parents to create children
        desired_length = len(population) - len(parents)
        children = []

        while len(children) < desired_length:
            male, female = rd.sample(parents, 2)
            if male.combinable(female):
                children.append(self.crossover(male=male, female=female))
        parents.extend(children)

        return parents

    def crossover(self, male : ImageIndividual, female : ImageIndividual):
        assert male.combinable(female)
        return male.combine(female)

    def mutate(self, individual: ImageIndividual):
        pos_to_mutate = rd.randint(0, len(individual) - 1)
        individual.mutate(pos_to_mutate)
        individual.classify(force_recomputation=True)  # recompute classification

    def run(self, initial_population_generator : PopulationGenerator, steps = 10):

        population = [individual for individual in initial_population_generator]

        print(population)

        fitness_history = [self.grade(population), ]
        for i in range(steps):
            print("------------------Grade: %f, Generation %s with %d individuals------------------" % (
                fitness_history[-1], str(i + 1), len(population)))
            population = self.evolve(population, retain=0.2)
            fitness_history.append(self.grade(population))
            print(population)

        print("------------------Finished Process Grade History------------------")

        print(fitness_history)

        print("------------------Final Population------------------")
        print(population)
