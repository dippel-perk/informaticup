from genetic.population_generator.population_generator import PopulationGenerator
from classifier.offline_classifier import OfflineClassifier
from genetic.basic_approach import BasicApproach
from genetic.image_individual import ImageIndividual
from road_sign_class_mapper import RoadSignClassMapper


class GeneticPopulationGenerator(PopulationGenerator):

    def __init__(self, size, class_id: int, population_generator: PopulationGenerator, mutation_rate=0.04, steps=20):
        super().__init__(size=size)
        self._class_id = class_id
        self._population_generator = population_generator
        self._mutation_rate = mutation_rate
        self._steps = steps

    def __iter__(self):
        classifier = OfflineClassifier()
        genetic = BasicApproach(classifier=classifier, class_to_optimize=str(self._class_id),
                                mutation_rate=self._mutation_rate)
        population, _ = genetic.run(initial_population_generator=self._population_generator, steps=self._steps, verbose=False)

        top_population = sorted(population, key=lambda individual: individual.classification.value_for_class(class_name=str(self._class_id)), reverse=True)[:self.size]

        for individual in top_population:
            yield ImageIndividual(image=individual.image)
