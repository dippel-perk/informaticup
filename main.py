from genetic.basic_approach import BasicApproach
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from classifier.online_classifier import OnlineClassifier
if __name__ == '__main__':

    classifier = OnlineClassifier()

    genetic = BasicApproach(classifier=classifier, class_to_optimize="Vorfahrt")

    population_generator = RandomPopulationGenerator(size=10)
    genetic.run(initial_population_generator=population_generator, steps=10)