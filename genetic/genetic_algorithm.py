import datetime
import os
import pathlib
import random as rd
import string
from typing import List, Callable

from tqdm import tqdm

from classifier.classifier import Classifier
from config.classifier_configuration import ClassifierConfiguration
from genetic.image_individual import ImageIndividual
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities
from utils.output_utilities import print_info, print_success, print_space, make_bold


class GeneticAlgorithm:
    """
    Base Class for a number of genetic algorithms. The class provides full functionality. All methods can
    be overwritten by potential sub classes do have an influence on the behaviour of the algorithm.
    """

    def __init__(self, classifier: Classifier, class_to_optimize: string, retain_rate: float = 0.2,
                 random_select_rate: float = 0.00, mutation_rate: float = 1.0, mutation_intensity=0.05,
                 save_generations: bool = True, output_dir: str = None,
                 pixel_mutation_function=ImageUtilities.mutate_pixels):
        self._class_to_optimize = class_to_optimize
        self._classifier = classifier
        self._reset_parameters()
        self._retain_rate = retain_rate
        self._random_select_rate = random_select_rate
        self._mutation_rate = mutation_rate
        self._mutation_intensity = mutation_intensity
        self._save_history = save_generations
        self._pixel_mutation_function = pixel_mutation_function

        saved_args = locals()
        if save_generations:
            if output_dir is None:
                self._output_dir = os.path.join('tmp', 'output', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            pathlib.Path(self._output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self._output_dir, 'config.txt'), 'w+') as f:
                for name, val in saved_args.items():
                    f.write('{}: {}\n'.format(name, val))

    @property
    def output_dir(self):
        return self._output_dir

    def _reset_parameters(self) -> None:
        """
        Resets the parameters of the genetic algorithm.
        :return: None
        """
        self._population_history = list()
        self._grade_history = list()

    def _get_latest_grade(self, convert_to_percent=True) -> float:
        """
        Returns the latest grade if present. If needed the grade will be convert into a percent value.
        :param convert_to_percent: Pass True if the conversion should be computed, False otherwise
        :return: The latest grade.
        """
        if len(self._grade_history) > 0:
            if convert_to_percent:
                return self._grade_history[-1] * 100
            else:
                return self._grade_history[-1]
        else:
            return -1

    def _initialize_with_population(self, population: List[ImageIndividual]) -> None:
        """
        Initializes the genetic algorithm with the given population.
        :param population: The initial population.
        :return: None
        """
        self._reset_parameters()
        self._population_history.append(population)
        self._compute_fitness_of_current_population()
        self._grade_history.append(self._grade())

    def _get_current_population(self) -> List[ImageIndividual]:
        """
        Returns the latest population.
        :return: The latest population.
        """
        assert len(self._population_history) > 0
        return self._population_history[-1]

    def _classify_individual(self, individual: ImageIndividual, force_recomputation: bool = False) -> None:
        """
        If not already done this method classifies a given individual and saves the result to the
        classification attribute of the object. If necessary one could force a recomputation even if an
        individual is already classified.
        :param individual: The individual which should be classified.
        :param force_recomputation: True if a reclassification should be forced.
        :return: None
        """
        if not individual.classification or force_recomputation:
            individual.classification = self._classifier.classify(individual.image)

    def _fitness(self, individual: ImageIndividual) -> float:
        """
        Returns the fitness of a given individual. The fitness is the confidence that the individual belongs
        to the target class.
        :param individual: The individual.
        :return: The fitness value.
        """
        self._classify_individual(individual)
        return individual.classification.value_for_class(self._class_to_optimize)

    def _compute_fitness_of_current_population(self) -> None:
        """
        Computes the fitness of the current population. Uses the classify batch function of the classifier, which
        can speed up the computation in certain classifiers.
        :return: None
        """
        population = self._get_current_population()
        classifications = self._classifier.classify_batch([individual.image for individual in population])
        for i in range(len(classifications)):
            population[i].classification = classifications[i]

    def _grade(self, population: List[ImageIndividual] = None) -> float:
        """
        Grades the given population. The grade of the population is the maximal fitness value
        over all individuals of the population. If not population is provided, the current population is used.
        :param population: The population.
        :return: The grade.
        """
        if not population:
            population = self._get_current_population()
        return max(self._fitness(x) for x in population)

    def _save_generation(self, step: int) -> None:
        """
        Saves the individuals of the generation to the output directory. The individuals are ordered by their
        fitness values.
        :param step: The step, i.e. subdirectory in which the files should be saved.
        :return: None
        """
        population = sorted(self._get_current_population(), key=lambda x: self._fitness(x), reverse=True)
        pathlib.Path(self._output_dir, str(step)).mkdir(parents=True, exist_ok=True)
        for idx, individual in enumerate(population):
            individual.image.save(
                os.path.join(self._output_dir, str(step),
                             '{}_{}.{}'.format(idx, self._fitness(individual),
                                               ClassifierConfiguration.DESIRED_IMAGE_EXTENSION)))

    def _evolve(self) -> None:
        """
        Evolves the current generation and appends the next generation to the population history and fitness history
        Calls _pre_evolve() and _post_evolve() before the process starts and after the process ends.
        When overwriting this method, make sure the generation was added and both methods were called.
        :return: None
        """

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
        self._compute_fitness_of_current_population()
        self._grade_history.append(self._grade())

    def _crossover(self, male: ImageIndividual, female: ImageIndividual) -> ImageIndividual:
        """
        Performs a basic pixel crossover method of two individuals. With the probability mutation_rate the
        child individual will get mutated as well.
        :param male: The first individual
        :param female: The second individual
        :return: A crossed over child individual
        """
        assert self._combinable(male, female)
        image = ImageUtilities.combine_images(male.image, female.image)
        individual = ImageIndividual(image=image)
        if self._mutation_rate > rd.random():
            individual = self._mutate(individual)
        return individual

    def _mutate(self, individual: ImageIndividual) -> ImageIndividual:
        """
        Mutates the given individual by changing a certain amount of pixels to certain values. The values are determined
        by the pixel mutation function of the object. The amount of pixels is related to the mutation_intensity
        attribute.
        :param individual: The individual which should be mutated.
        :return: The mutated individual.
        """
        pixels = rd.sample(list(range(len(individual))),
                           min(int(len(individual) * self._mutation_intensity), len(individual)))
        self._pixel_mutation_function(individual.image, pixels)
        return individual

    def _combinable(self, male: ImageIndividual, female: ImageIndividual) -> bool:
        """
        Decides if two given individuals are combinable Two individuals are combinable if they share at least one
        class of their classifications.
        :param male: The first individual.
        :param female: The second individual.
        :return: True if the individuals are combinable, False otherwise.
        """
        return male.classification.share_classes(female.classification)

    def run(self,
            initial_population_generator: PopulationGenerator,
            grade_limit=2.0,
            steps=100,
            verbose=True,
            on_generation_evolved: Callable[[None], None] = None):
        """
        Runs the genetic algorithm.
        :param initial_population_generator: The generator which should be used to generate the initial population
        :param grade_limit: The grade limit. The algorithm stops when a generation reaches the given grade.
        :param steps: The maximum number of generations
        :param verbose: Specifies if the algorithm should output information during the simulation.
        :param on_generation_evolved: A method which gets called when a new generation is evolved.
        :return: The last generation, i.e. the generation which maximizes the grade.
        """

        initial_population_progress_bar = None
        if verbose:
            if self._save_history:
                print_info("The complete generation history will be saved to the output directory %s" % self.output_dir)
                print_space()
            print_info("Generating initial population with %s" % make_bold(initial_population_generator.__repr__()))

            initial_population_progress_bar = tqdm(total=initial_population_generator.progress_bar_total)
            initial_population_progress_bar.set_description(initial_population_generator.progress_bar_description)
            initial_population_generator.register_progress_bar(progress_bar=initial_population_progress_bar)

        self._initialize_with_population([individual for individual in initial_population_generator])

        if verbose:
            print_success("Initial population generated successful.")
            initial_population_progress_bar.close()
            print_info("Starting the evolution process")

        needed_steps = steps

        for i in range(1, steps + 1):

            if self._save_history:
                self._save_generation(step=i)

            if verbose:
                print("[%s] generation with grade: %f%%" % (str(i).zfill(3), self._get_latest_grade()))

            if self._grade_history[-1] >= grade_limit:

                if verbose:
                    print_info("Generation reached the grade limit of %d%%" % (grade_limit * 100))

                needed_steps = i
                break

            self._evolve()

            if on_generation_evolved is not None:
                on_generation_evolved()

        if verbose:
            print_success("Finished on generation %d with grade %f%%" % (needed_steps, self._get_latest_grade()))

        return self._get_current_population(), needed_steps

    def __repr__(self):
        return "Genetic Algorithm"