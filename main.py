import argparse

from genetic.basic_approach import BasicApproach
from genetic.population_generator.population_generator import PopulationGenerator
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.population_generator.train_color_population_generator import TrainColorPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import SampleImagesRearrangePopulationGenerator
from classifier.online_classifier import OnlineClassifier
from road_sign_class_mapper import RoadSignClassMapper

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rand', action='store_true')
    group.add_argument('--color', action='store_true')
    group.add_argument('--sample', action='store_true')

    args = parser.parse_args()

    classifier = OnlineClassifier()

    class_name = "Zulässige Höchstgeschwindigkeit (30)"
    class_id = RoadSignClassMapper().get_class_by_name(name=class_name)

    genetic = BasicApproach(classifier=classifier, class_to_optimize=class_name, mutation_rate=0.1)

    if args.color:
        population_generator = TrainColorPopulationGenerator(size=10, target_class=class_id, image_dir='../GTSRB/Final_Training/')
    elif args.rand:
        population_generator = RandomPopulationGenerator(size=10)
    elif args.sample:
        population_generator = SampleImagesRearrangePopulationGenerator(size=10, target_class=class_id, image_dir='../GTSRB/Final_Training/')
    else:
        population_generator = PopulationGenerator(size=10)

    genetic.run(initial_population_generator=population_generator, steps=10)