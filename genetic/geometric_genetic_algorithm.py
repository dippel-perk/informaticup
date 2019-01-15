import random as rd
import string
from typing import Callable

from classifier.classifier import Classifier
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.geometric.geometric_individual import GeometricIndividual
from genetic.geometric.geometric_objects import GeometricObject
from genetic.image_individual import ImageIndividual


class GeometricGeneticAlgorithm(GeneticAlgorithm):
    """
    Genetic algorithm modification so that the algorithm does work on geometric individuals.
    """

    def __init__(self, classifier: Classifier, class_to_optimize: string,
                 mutation_function: Callable[[GeometricObject], GeometricObject], retain_rate: float = 0.2,
                 random_select_rate: float = 0.00, mutation_rate: float = 1.0, mutation_intensity=0.05,
                 save_generations: bool = True, output_dir: str = None):
        super().__init__(classifier=classifier,
                         class_to_optimize=class_to_optimize,
                         retain_rate=retain_rate,
                         random_select_rate=random_select_rate,
                         mutation_rate=mutation_rate,
                         mutation_intensity=mutation_intensity, save_generations=save_generations,
                         output_dir=output_dir)

        self._mutation_callback = mutation_function

    def _crossover(self, male: GeometricIndividual, female: GeometricIndividual):
        """
        Performs a basic geometric crossover of two geometric individuals. With the probability mutation_rate the
        child individual will get mutated as well.
        :param male: The first individual
        :param female: The second individual
        :return: A crossed over child individual
        """
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

        child = GeometricIndividual(new_objects)

        if self._mutation_rate > rd.random():
            child = self._mutate(individual=child)

        return child

    def _mutate(self, individual: ImageIndividual):
        """
        Mutates the given individual by changing a certain amount of geometric objects. The objects are mutated using
        the mutation callback method of this object. The amount of objects is related to the mutation_intensity
        attribute.
        :param individual: The individual which should be mutated.
        :return: The mutated individual.
        """
        if individual.__class__ == GeometricIndividual:
            new_objects = individual.get_objects()

            for i in range(len(new_objects)):
                if self._mutation_intensity > rd.random():
                    new_objects[i] = self._mutation_callback(new_objects[i])

            return GeometricIndividual(new_objects)
        else:
            raise NotImplementedError

    def __repr__(self):
        return "Geometric Genetic Algorithm"