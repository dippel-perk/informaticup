import argparse

from genetic.basic_approach import BasicApproach
from genetic.population_generator.population_generator import PopulationGenerator
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.population_generator.train_color_population_generator import TrainColorPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import \
    SampleImagesRearrangePopulationGenerator
from genetic.population_generator.gradient_population_generator import GradientPopulationGenerator
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from genetic.population_generator.random_brute_force_population_generator import RandomBruteForcePopulationGenerator
from classifier.online_classifier import OnlineClassifier
from road_sign_class_mapper import RoadSignClassMapper
from genetic.geometric.circle_population_generator import CirclePopulationGenerator
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.geometric.polygon_population_generator import PolygonPopulationGenerator
from genetic.geometric.bitmap_population_generator import BitmapPopulationGenerator
from PIL import Image
import PIL.ImageOps

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confidence', required=True, type=float)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rand', action='store_true')
    group.add_argument('--color', action='store_true')
    group.add_argument('--sample', action='store_true')
    group.add_argument('--genetic', action='store_true')
    group.add_argument('--gradient', action='store_true')
    group.add_argument('--brute-force', action='store_true')
    group.add_argument('--circle', action='store_true')
    group.add_argument('--polygon', action='store_true')
    group.add_argument('--gilogo', action='store_true')

    args = parser.parse_args()

    classifier = OnlineClassifier()

    class_name = "Vorfahrt"
    class_id = RoadSignClassMapper().get_class_by_name(name=class_name)

    genetic = BasicApproach(classifier=classifier, class_to_optimize=class_name, mutation_intensity=0.05)

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
    elif args.gradient:
        population_generator = GradientPopulationGenerator(size=10, class_id=class_id,
                                                           population_generator=TrainColorPopulationGenerator(size=50,
                                                                                                              target_class=class_id,
                                                                                                              image_dir=image_path))
    elif args.genetic:
        population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=50,
                                                          population_generator=SampleImagesRearrangePopulationGenerator(size=100, target_class=class_id,
                                                                        image_dir=image_path))
    elif args.brute_force:
        population_generator = RandomBruteForcePopulationGenerator(size= size, classifier=classifier, target_class=class_name)
    elif args.circle:
        population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=30,
                                                          population_generator=CirclePopulationGenerator(100), algorithm=GeometricGeneticAlgorithm, mutation_intensity=0.05)
        genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name, mutation_intensity=0.1)
    elif args.polygon:
        population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=20,
                                                          population_generator=PolygonPopulationGenerator(100),
                                                          algorithm=GeometricGeneticAlgorithm, mutation_intensity=0.05)
        genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name, mutation_intensity=0.1)
    elif args.gilogo:
        image = Image.open("gi-logo.jpg")
        inverted_image = PIL.ImageOps.invert(image)
        image = inverted_image.convert("1").resize((200, 200))
        population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=20,
                                                          population_generator=BitmapPopulationGenerator(100, image),
                                                          algorithm=GeometricGeneticAlgorithm, mutation_intensity=0.05)
        genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name, mutation_intensity=0.1)
    else:
        population_generator = PopulationGenerator(size=size)

    population, _ = genetic.run(initial_population_generator=population_generator, grade_limit=args.confidence, steps=40)

    best =  max(population, key=lambda individual: individual.classification.value_for_class(class_name=class_name))
    best.image.save("tmp/best_" + str(class_id) + '.png')

    print(OnlineClassifier.SEEN_CLASSES)
