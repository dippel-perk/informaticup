from classifier.classifier import Classifier
from genetic.image_individual import ImageIndividual
from typing import List
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities
import string

import random as rd
import time


class GeneticAlgorithm:
    """
    Base Class for a number of genetic algorithms. The class provides full functionality. All methods can
    be overwritten by potential sub classes do have an influence on the behaviour of the algorithm.
    """

    def __init__(self, classifier: Classifier, class_to_optimize: string, retain_rate: float = 0.2,
                 random_select_rate: float = 0.00, mutation_rate: float = 1.0, mutation_intensity = 0.05):
        self._class_to_optimize = class_to_optimize
        self._classifier = classifier
        self._reset_parameters()
        self._retain_rate = retain_rate
        self._random_select_rate = random_select_rate
        self._mutation_rate = mutation_rate
        self._mutation_intensity = mutation_intensity

    def _reset_parameters(self):
        self._population_history = list()
        self._fitness_history = list()

    def _initialize_with_population(self, population: List[ImageIndividual]):
        self._reset_parameters()
        self._population_history.append(population)
        self._compute_fitness()
        self._fitness_history.append(self._grade())

    def _get_current_population(self):
        return self._population_history[-1]

    def _classify_individual(self, individual: ImageIndividual, force_recomputation=False):
        if not individual.classification or force_recomputation:
            individual.classification = self._classifier.classify(individual.image)

    def _fitness(self, individual: ImageIndividual):
        self._classify_individual(individual)
        return individual.classification.value_for_class(self._class_to_optimize)

    def _compute_fitness(self):
        population = self._get_current_population()
        classifications = self._classifier.classify_batch([individual.image for individual in population])
        for i in range(len(classifications)):
            population[i].classification = classifications[i]

    def _grade(self, population: List[ImageIndividual] = None):
        if not population:
            population = self._get_current_population()
        return max(self._fitness(x) for x in population)

    def _pre_evolve(self) -> None:
        """
        Method should be called before the current population is evolved.
        :return: None
        """
        pass

    def _post_evolve(self) -> None:
        """
        Method should be called after the current population is evolved.
        :return: None
        """
        pass

    def _evolve(self) -> None:
        """
        Evolves the current generation and appends the next generation to the population history and fitness history
        Calls _pre_evolve() and _post_evolve() before the process starts and after the process ends.
        When overwriting this method, make sure the generation was added and both methods were called.
        :return: None
        """

        assert all(individual.image for individual in self._get_current_population())

        self._pre_evolve()

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
        self._compute_fitness()
        self._fitness_history.append(self._grade())

        self._post_evolve()

    def _crossover(self, male: ImageIndividual, female: ImageIndividual):
        assert self._combinable(male, female)
        image = ImageUtilities.combine_images(male.image, female.image)
        individual = ImageIndividual(image=image)
        if self._mutation_rate > rd.random():
            individual = self._mutate(individual)
        return individual

    def _mutate(self, individual: ImageIndividual):
        pixels = rd.sample(list(range(len(individual))), min(int(len(individual) * self._mutation_intensity), len(individual)))
        ImageUtilities.mutate_pixels(individual.image, pixels)
        return individual

    def _combinable(self, male, female):
        #TODO: The combinable method might not be the best choice
        return male.classification.share_classes(female.classification)

    def run(self, initial_population_generator: PopulationGenerator, grade_limit = 2.0, steps=100, verbose=True):
        """
        Runs the genetic algorithm
        :param initial_population_generator: The generator which should be used to generate the initial population
        :param grade_limit: The grade limit. The algorithm stops when a generation reaches the given grade.
        :param steps: The maximum number of generations
        :param verbose: Specifies if the algorithm should output information during the simulation.
        :return: The last generation, i.e. the generation which maximizes the grade.
        """

        self._initialize_with_population([individual for individual in initial_population_generator])

        needed_steps = steps

        for i in range(steps):

            if verbose:
                print("[%s] Generation: Grade: %f" % (str(i + 1).zfill(3), self._fitness_history[-1]))

            if self._fitness_history[-1] >= grade_limit:
                needed_steps = i
                break

            self._evolve()

        if verbose:
            print("[END] Genetic Algorithm terminated")
            print(self._fitness_history)
            print(self._get_current_population())

        return self._get_current_population(), needed_steps