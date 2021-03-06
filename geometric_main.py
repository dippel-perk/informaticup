import argparse

from classifier.online_classifier import OnlineClassifier
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from genetic.population_generator.population_generator import PopulationGenerator
from genetic.population_generator.random_brute_force_population_generator import RandomBruteForcePopulationGenerator
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import \
    SampleImagesRearrangePopulationGenerator
from genetic.population_generator.train_color_population_generator import TrainColorPopulationGenerator
from utils.road_sign_class_mapper_utilities import RoadSignClassMapper

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confidence', required=True, type=float)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rand', action='store_true')
    group.add_argument('--color', action='store_true')
    group.add_argument('--sample', action='store_true')
    group.add_argument('--genetic', action='store_true')
    group.add_argument('--brute-force', action='store_true')

    args = parser.parse_args()

    classifier = OnlineClassifier()

    class_name = "Fußgänger"
    class_id = RoadSignClassMapper().get_class_by_name(name=class_name)

    genetic = GeneticAlgorithm(classifier=classifier, class_to_optimize=class_name, mutation_intensity=0.05)

    image_path = '../GTSRB/Final_Training/Images'
    size = 20

    if args.color:
        population_generator = TrainColorPopulationGenerator(size=size, target_class=class_id,
                                                             image_dir=image_path)
    elif args.rand:
        population_generator = RandomPopulationGenerator(size=size)
    elif args.sample:
        population_generator = SampleImagesRearrangePopulationGenerator(size=size, target_class=class_id,
                                                                        image_dir=image_path)
    elif args.genetic:
        population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=50,
                                                          population_generator=SampleImagesRearrangePopulationGenerator(
                                                              size=100, target_class=class_id,
                                                              image_dir=image_path))
    elif args.brute_force:
        population_generator = RandomBruteForcePopulationGenerator(size=size, classifier=classifier,
                                                                   target_class=class_name)
    else:
        population_generator = PopulationGenerator(size=size)

    genetic.run(initial_population_generator=population_generator, grade_limit=args.confidence, steps=200)
