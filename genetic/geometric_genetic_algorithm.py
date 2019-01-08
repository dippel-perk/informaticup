from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.basic_approach import BasicApproach
from genetic.image_individual import ImageIndividual
from classifier.classifier import Classifier
from utils.image_utilities import ImageUtilities
from genetic.geometric.geometric_individual import GeometricIndividual
from genetic.geometric.circle_population_generator import generate_circle
import random as rd
import string


class GeometricGeneticAlgorithm(BasicApproach):

    def __init__(self, classifier: Classifier, class_to_optimize: string, retain_rate: float = 0.2,
                 random_select_rate: float = 0.00, mutation_rate: float = 1.0, mutation_intensity=0.05):
        super().__init__(classifier=classifier,
                         class_to_optimize=class_to_optimize,
                         retain_rate=retain_rate,
                         random_select_rate=random_select_rate,
                         mutation_rate=mutation_rate,
                         mutation_intensity=mutation_intensity)

    def _crossover(self, male: GeometricIndividual, female: GeometricIndividual):
        assert self._combinable(male, female)
        new_objects = []

        male_objects = male.get_objects()
        female_objects = female.get_objects()
        parent_objects = [male_objects, female_objects]
        average = int((len(male_objects) + len(female_objects)) / 2)

        for i in range(average):
            idx = rd.choice([0, 1])
            if i >= len(parent_objects[idx]):
                idx = (idx + 1) % 2
            obj = parent_objects[idx][i]
            new_objects.append(obj)

        if self._mutation_rate > rd.random():
            for i in range(len(new_objects)):
                if self._mutation_intensity > rd.random():
                    new_objects[i] = generate_circle(10, 5)
        return GeometricIndividual(new_objects)

    def _mutate(self, individual: ImageIndividual):
        raise NotImplementedError
