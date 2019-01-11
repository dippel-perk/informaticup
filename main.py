import argparse

import PIL.ImageOps
from PIL import Image

from classifier.online_classifier import OnlineClassifier
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.geometric.geometric_mutations import GeometricMutations
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from genetic.population_generator.geometric.bitmap_population_generator import BitmapPopulationGenerator
from genetic.population_generator.geometric.circle_population_generator import CirclePopulationGenerator
from genetic.population_generator.geometric.polygon_population_generator import PolygonPopulationGenerator
from genetic.population_generator.gradient_population_generator import GradientPopulationGenerator
from genetic.population_generator.population_generator import PopulationGenerator
from genetic.population_generator.random_brute_force_population_generator import RandomBruteForcePopulationGenerator
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import \
    SampleImagesRearrangePopulationGenerator
from genetic.population_generator.train_color_population_generator import TrainColorPopulationGenerator
from genetic.population_generator.geometric.tile_population_generator import TilePopulationGenerator
from road_sign_class_mapper import RoadSignClassMapper

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
    group.add_argument('--tiles', action='store_true')

    args = parser.parse_args()

    classifier = OnlineClassifier()

    class_name = "Zulässige Höchstgeschwindigkeit (70)"
    class_id = RoadSignClassMapper().get_class_by_name(name=class_name)

    genetic = GeneticAlgorithm(classifier=classifier, class_to_optimize=class_name, mutation_intensity=0.05)

    image_path = '../GTSRB/Final_Training/Images'
    size = 20

    if args.color:
        population_generator = TrainColorPopulationGenerator(size=size,
                                                             target_class=class_id,
                                                             image_dir=image_path)
    elif args.rand:
        population_generator = RandomPopulationGenerator(size=size)
    elif args.sample:
        population_generator = SampleImagesRearrangePopulationGenerator(size=size,
                                                                        target_class=class_id,
                                                                        image_dir=image_path)
    elif args.gradient:
        population_generator = GradientPopulationGenerator(size=10,
                                                           class_id=class_id,
                                                           population_generator=
                                                           TrainColorPopulationGenerator(size=50,
                                                                                         target_class=class_id,
                                                                                         image_dir=image_path)
                                                           )
    elif args.genetic:
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=50,
                                                          population_generator=
                                                          SampleImagesRearrangePopulationGenerator(size=100,
                                                                                                   target_class=class_id,
                                                                                                   image_dir=image_path)
                                                          )
    elif args.brute_force:
        population_generator = RandomBruteForcePopulationGenerator(size=size,
                                                                   classifier=classifier,
                                                                   target_class=class_name)
    elif args.circle:
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=30,
                                                          population_generator=CirclePopulationGenerator(100),
                                                          algorithm=GeometricGeneticAlgorithm,
                                                          mutation_intensity=0.05,
                                                          mutation_function=GeometricMutations.mutate_circle_function())
        genetic = GeometricGeneticAlgorithm(classifier=classifier,
                                            class_to_optimize=class_name,
                                            mutation_intensity=0.1,
                                            mutation_function=GeometricMutations.mutate_circle_function())
    elif args.polygon:
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=20,
                                                          population_generator=PolygonPopulationGenerator(100),
                                                          algorithm=GeometricGeneticAlgorithm,
                                                          mutation_intensity=0.05,
                                                          mutation_function=GeometricMutations.mutate_polygon_function(
                                                              n=3))
        genetic = GeometricGeneticAlgorithm(classifier=classifier,
                                            class_to_optimize=class_name,
                                            mutation_intensity=0.1,
                                            mutation_function=GeometricMutations.mutate_polygon_function(n=3)
                                            )
    elif args.gilogo:
        image = Image.open("gi-logo.jpg")
        inverted_image = PIL.ImageOps.invert(image)
        image = inverted_image.convert("1").resize((200, 200))
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=20,
                                                          population_generator=BitmapPopulationGenerator(100,
                                                                                                         image,
                                                                                                         avg_num=50),
                                                          algorithm=GeometricGeneticAlgorithm,
                                                          mutation_intensity=0.05,
                                                          mutation_function=GeometricMutations.mutate_bitmap_function(
                                                              img=image))
        genetic = GeometricGeneticAlgorithm(classifier=classifier,
                                            class_to_optimize=class_name,
                                            mutation_intensity=0.1,
                                            mutation_function=GeometricMutations.mutate_bitmap_function(img=image)
                                            )
    elif args.tiles:
        color1 = (255, 224, 130)
        color2 = (255, 160, 0)
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=20,
                                                          population_generator=TilePopulationGenerator(100,
                                                                                                       color1=color1,
                                                                                                       color2=color2),
                                                          algorithm=GeometricGeneticAlgorithm,
                                                          mutation_intensity=0.05,
                                                          mutation_function=GeometricMutations.mutate_tile_function(
                                                              color1, color2))
        genetic = GeometricGeneticAlgorithm(classifier=classifier,
                                            class_to_optimize=class_name,
                                            mutation_intensity=0.1,
                                            mutation_function=GeometricMutations.mutate_tile_function(color1, color2)
                                            )



    else:
        population_generator = PopulationGenerator(size=size)

    population, _ = genetic.run(initial_population_generator=population_generator,
                                grade_limit=args.confidence,
                                steps=40)

    best = max(population, key=lambda individual: individual.classification.value_for_class(class_name=class_name))
    best.image.save("tmp/best_" + str(class_id) + '.png')
