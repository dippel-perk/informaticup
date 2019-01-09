from genetic.population_generator.population_generator import PopulationGenerator
from classifier.offline_classifier import OfflineClassifier
from genetic.grade_average_genetic_algorithm import GradeAverageGeneticAlgorithm
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.geometric.geometric_mutations import GeometricMutations

class GeneticPopulationGenerator(PopulationGenerator):

    def __init__(self, size, class_id: int, population_generator: PopulationGenerator, mutation_intensity=0.1, steps=20, algorithm=GradeAverageGeneticAlgorithm):
        super().__init__(size=size)
        self._class_id = class_id
        self._population_generator = population_generator
        self._mutation_intensity = mutation_intensity
        self._steps = steps
        self._algorithm = algorithm

    def __iter__(self):
        classifier = OfflineClassifier()

        if self._algorithm == GeometricGeneticAlgorithm:
            genetic = self._algorithm(classifier=classifier, class_to_optimize=str(self._class_id),
                                      mutation_intensity=self._mutation_intensity, mutation_function=GeometricMutations.mutate_circle_function())
        else:
            genetic = self._algorithm(classifier=classifier, class_to_optimize=str(self._class_id),
                                mutation_intensity=self._mutation_intensity)
        population, _ = genetic.run(initial_population_generator=self._population_generator, steps=self._steps, grade_limit=1.1, verbose=True)

        top_population = sorted(population, key=lambda individual: individual.classification.value_for_class(class_name=str(self._class_id)), reverse=True)[:self.size]

        print("[Info] Genetic population generator's final population was graded as follows by the offline classifier", [individual.classification.value_for_class(class_name=str(self._class_id)) for individual in top_population])

        for individual in top_population:
            individual.classification = None
            yield individual
