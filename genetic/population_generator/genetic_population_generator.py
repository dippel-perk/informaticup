from classifier.offline_classifier import OfflineClassifier
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.grade_average_genetic_algorithm import GradeAverageGeneticAlgorithm
from genetic.population_generator.population_generator import PopulationGenerator
from utils.image_utilities import ImageUtilities
from utils.output_utilities import print_info


class GeneticPopulationGenerator(PopulationGenerator):
    """
    Generates a population which performs well on a local substitute network.
    """

    def __init__(self, size, class_id: int, population_generator: PopulationGenerator, mutation_intensity=0.1, steps=20,
                 algorithm=GradeAverageGeneticAlgorithm, mutation_function=None,
                 pixel_mutation_function=ImageUtilities.mutate_pixels):
        super().__init__(size=size)
        self._class_id = class_id
        self._population_generator = population_generator
        self._mutation_intensity = mutation_intensity
        self._steps = steps
        self._algorithm = algorithm
        self._mutation_function = mutation_function
        self._pixel_mutation_function = pixel_mutation_function

    @property
    def progress_bar_total(self) -> int:
        """
        Returns the number of steps needed to fill a possible progess bar.
        :return: The number of steps.
        """
        return self._steps

    @property
    def progress_bar_description(self) -> str:
        """
        Returns the description of the progress bar.
        :return: The description
        """
        return "Steps"

    def __iter__(self):
        """
        This function is called when the population generator is iterated. It yields a set image individuals.
        :return: Yields image individuals
        """
        classifier = OfflineClassifier()

        if self._algorithm == GeometricGeneticAlgorithm:
            genetic = self._algorithm(classifier=classifier, class_to_optimize=str(self._class_id),
                                      mutation_intensity=self._mutation_intensity,
                                      mutation_function=self._mutation_function, save_generations=False)
        elif self._algorithm == GeneticAlgorithm:
            genetic = self._algorithm(classifier=classifier, class_to_optimize=str(self._class_id),
                                      mutation_intensity=self._mutation_intensity, save_generations=False,
                                      pixel_mutation_function=self._pixel_mutation_function)
        elif self._algorithm == GradeAverageGeneticAlgorithm:
            genetic = self._algorithm(classifier=classifier, class_to_optimize=str(self._class_id),
                                      mutation_intensity=self._mutation_intensity, save_generations=False,
                                      pixel_mutation_function=self._pixel_mutation_function)
        else:
            raise NotImplementedError("Genetic population generator does not recognize the given genetic algorithm.")

        def on_generation_evolved():
            self._progress_bar_step()

        population, _ = genetic.run(initial_population_generator=self._population_generator,
                                    steps=self._steps,
                                    verbose=False,
                                    on_generation_evolved=on_generation_evolved)

        top_population = sorted(population, key=lambda individual: individual.classification.value_for_class(
            class_name=str(self._class_id)), reverse=True)[:self.size]

        average_fitness = sum(
            (individual.classification.value_for_class(class_name=str(self._class_id)) for individual in
             top_population)) / self.size

        print_info("The generated population has an average fitness of %f%% on the substitute network" % (
        average_fitness * 100))

        for individual in top_population:
            individual.classification = None
            yield individual

    def __repr__(self):
        return "Genetic Population Generator using %s" % self._population_generator.__repr__()
