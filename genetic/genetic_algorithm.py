from classifier.classifier import Classifier
from genetic.image_individual import ImageIndividual
from typing import List
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities
import string

import random as rd

class GeneticAlgorithm:

    def __init__(self, classifier: Classifier, class_to_optimize: string):
        self._class_to_optimize = class_to_optimize
        self._classifier = classifier
        self._reset_parameters()

    def _reset_parameters(self):
        self._population_history = list()
        self._fitness_history = list()

    def _initialize_with_population(self, population: List[ImageIndividual]):
        self._reset_parameters()
        self._population_history.append(population)
        self._fitness_history.append(self._grade())

    def _get_current_population(self):
        return self._population_history[-1]

    def _classify_individual(self, individual: ImageIndividual, force_recomputation = False):
        if not individual.classification or force_recomputation:
            file = ImageUtilities.save_image_to_tempfile(individual.image)
            individual.classification = self._classifier.classify(file)

    def _fitness(self, individual : ImageIndividual):
        self._classify_individual(individual)
        return individual.classification.value_for_class(self._class_to_optimize)

    def _grade(self, population: List[ImageIndividual] = None):
        if not population:
            population = self._get_current_population()
        summed = sum(self._fitness(x) for x in population)
        return summed / len(population)

    def _evolve(self, retain: float = 0.2, random_select: float = 0.05, mutate: float = 0.01):

        current_population = self._get_current_population()

        graded = sorted(current_population, key=lambda x: self._fitness(x), reverse=True)

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
                self._mutate(individual=individual)

        # crossover parents to create children
        desired_length = len(current_population) - len(parents)
        children = []

        while len(children) < desired_length:
            male, female = rd.sample(parents, 2)
            if self._combinable(male, female):
                children.append(self._crossover(male=male, female=female))
        parents.extend(children)

        self._population_history.append(parents)
        self._fitness_history.append(self._grade())


    def _crossover(self, male : ImageIndividual, female : ImageIndividual):
        assert self._combinable(male, female)
        image = ImageUtilities.combine_images(male.image, female.image)
        return ImageIndividual(image=image)

    def _mutate(self, individual: ImageIndividual):
        pos_to_mutate = rd.randint(0, len(individual) - 1)
        ImageUtilities.mutate_pixel(individual.image, pos_to_mutate)
        self._classify_individual(individual, force_recomputation=True)  # recompute classification

    def _combinable(self, male, female):
        return male.classification.share_classes(female.classification)

    def run(self, initial_population_generator : PopulationGenerator, steps = 10):

        self._initialize_with_population([individual for individual in initial_population_generator])

        print(self._get_current_population())
        for i in range(steps):
            print("------------------Grade: %f, Generation %s with %d individuals------------------" %
                  (self._fitness_history[-1], str(i + 1), len(self._get_current_population())))

            self._evolve(retain=0.2)

            print(self._get_current_population())

        print("------------------Finished Process Grade History------------------")

        print(self._fitness_history)

        print("------------------Final Population------------------")
        print(self._get_current_population())
