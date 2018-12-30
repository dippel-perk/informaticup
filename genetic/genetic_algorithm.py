from classifier.classifier import Classifier
from genetic.image_individual import ImageIndividual
from typing import List
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities
import string

import random as rd


class GeneticAlgorithm:
    """
    Base Class for a number of genetic algorithms. The class provides full functionality. All methods which can
    be overwritten by potential sub classes do have an influence on the behaviour of the algorithm.
    """

    def __init__(self, classifier: Classifier, class_to_optimize: string, retain_rate: float = 0.2,
                 random_select_rate: float = 0.05, mutation_rate: float = 0.01):
        self._class_to_optimize = class_to_optimize
        self._classifier = classifier
        self._reset_parameters()
        self._retain_rate = retain_rate
        self._random_select_rate = random_select_rate
        self._mutation_rate = mutation_rate

    def _reset_parameters(self):
        self._population_history = list()
        self._fitness_history = list()

    def _initialize_with_population(self, population: List[ImageIndividual]):
        self._reset_parameters()
        self._population_history.append(population)
        self._fitness_history.append(self._grade())

    def _get_current_population(self):
        return self._population_history[-1]

    def _classify_individual(self, individual: ImageIndividual, force_recomputation=False):
        if not individual.classification or force_recomputation:
            file = ImageUtilities.save_image_to_tempfile(individual.image)
            individual.classification = self._classifier.classify(file)

    def _fitness(self, individual: ImageIndividual):
        self._classify_individual(individual)
        return individual.classification.value_for_class(self._class_to_optimize)

    def _grade(self, population: List[ImageIndividual] = None):
        if not population:
            population = self._get_current_population()
        summed = sum(self._fitness(x) for x in population)
        return summed / len(population)

    def _evolve(self):

        assert all(individual.image for individual in self._get_current_population())

        current_population = self._get_current_population()

        graded = sorted(current_population, key=lambda x: self._fitness(x), reverse=True)

        retain_length = int(len(graded) * self._retain_rate)
        parents = graded[:retain_length]

        # randomly add other individuals to
        # promote genetic diversity
        for individual in graded[retain_length:]:
            if self._random_select_rate > rd.random():
                parents.append(individual)

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

    def _crossover(self, male: ImageIndividual, female: ImageIndividual):
        assert self._combinable(male, female)

        # mutate some individuals before crossover
        if self._mutation_rate > rd.random():
            male = self._mutate(male)
        if self._mutation_rate > rd.random():
            female = self._mutate(female)

        image = ImageUtilities.combine_images(male.image, female.image)
        return ImageIndividual(image=image)

    def _mutate(self, individual: ImageIndividual):
        image = individual.image
        pixels = rd.sample(list(range(len(individual))), int(len(individual) * self._mutation_rate))
        ImageUtilities.mutate_pixels(image, pixels)
        return ImageIndividual(image=image)

    def _combinable(self, male, female):
        return male.classification.share_classes(female.classification)

    def run(self, initial_population_generator: PopulationGenerator, grade_limit=2.0, steps=100, verbose=True):

        self._initialize_with_population([individual for individual in initial_population_generator])

        for i in range(steps):

            if verbose:
                print("[%s] Generation; Grade: %f" % (str(i + 1), self._fitness_history[-1]))

            if self._fitness_history[-1] > grade_limit:
                break

            self._evolve()


        if verbose:
            print("Finshed")
            print(self._fitness_history)
            print(self._get_current_population())

        return self._get_current_population()