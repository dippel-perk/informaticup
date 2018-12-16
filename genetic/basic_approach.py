from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.image_individual import ImageIndividual
from image_utilities import ImageUtilities
from random import randint, random, sample


class BasicApproach(GeneticAlgorithm):

    def individual(self):
        individual = ImageIndividual(api=self.api, image=ImageUtilities.generate_random_noise())
        individual.classify()

        while individual.classification.value_for_class(self.class_to_optimize) == 0:
            individual = ImageIndividual(api=self.api, image=ImageUtilities.generate_random_noise())
            individual.classify()

        return individual

    def fitness(self, individual):
        individual.classify()
        return individual.classification.value_for_class(self.class_to_optimize)

    def grade(self, population):
        return max(self.fitness(x) for x in population)

    def evolve(self, population, retain=0.2, random_select=0.05, mutate=0.01):
        graded = sorted(population, key=lambda x: self.fitness(x), reverse=True)

        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]

        # randomly add other individuals to
        # promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if mutate > random():
                pos_to_mutate = randint(0, len(individual) - 1)
                individual.mutate(pos_to_mutate)
                individual.classify(force_recomputation=True)  # recompute classification

        # crossover parents to create children
        desired_length = len(population) - len(parents)
        children = []

        while len(children) < desired_length:
            male, female = sample(parents, 2)

            if male.combinable(female):
                children.append(male.combine(female))

        parents.extend(children)

        return parents
