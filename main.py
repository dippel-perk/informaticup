import argparse

import PIL.ImageOps
from PIL import Image
from genetic.population_generator.bitmap_population_generator import BitmapPopulationGenerator
from genetic.population_generator.polygon_population_generator import PolygonPopulationGenerator
from genetic.population_generator.tile_population_generator import TilePopulationGenerator

from classifier.online_classifier import OnlineClassifier
from config.program_arguments_configuration import ProgramArgumentsConfiguration
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.geometric.geometric_mutations import GeometricMutations
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.population_generator.circle_population_generator import CirclePopulationGenerator
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from genetic.population_generator.population_generator import PopulationGenerator
from genetic.population_generator.random_brute_force_population_generator import RandomBruteForcePopulationGenerator
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import \
    SampleImagesRearrangePopulationGenerator
from genetic.population_generator.train_color_population_generator import TrainColorPopulationGenerator

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=ProgramArgumentsConfiguration.PROGRAM_DESCRIPTION)

    parser.add_argument('-t', '--target',
                        required=True,
                        type=ProgramArgumentsConfiguration.target_class,
                        help=ProgramArgumentsConfiguration.TARGET_CLASS_DESCRIPTION)

    parser.add_argument('-c', '--confidence',
                        required=False,
                        default=2.0,
                        type=ProgramArgumentsConfiguration.percentage_float,
                        help=ProgramArgumentsConfiguration.CONFIDENCE_DESCRIPTION)
    parser.add_argument('-s', '--steps',
                        required=False,
                        type=int,
                        help=ProgramArgumentsConfiguration.STEPS_DESCRIPTION)

    parser.add_argument('--population-size',
                        required=False,
                        default=20,
                        type=int,
                        help=ProgramArgumentsConfiguration.POPULATION_SIZE_DESCRIPTION)

    parser.add_argument('--retain-rate',
                        required=False,
                        default=0.2,
                        type=ProgramArgumentsConfiguration.percentage_float,
                        help=ProgramArgumentsConfiguration.MUTATION_RATE_DESCRIPTION)
    parser.add_argument('--mutation-rate',
                        required=False,
                        default=1.0,
                        type=ProgramArgumentsConfiguration.percentage_float,
                        help=ProgramArgumentsConfiguration.MUTATION_RATE_DESCRIPTION)
    parser.add_argument('--mutation-intensity',
                        required=False,
                        default=0.05,
                        type=ProgramArgumentsConfiguration.percentage_float,
                        help=ProgramArgumentsConfiguration.MUTATION_INTENSITY_DESCRIPTION)
    parser.add_argument('--random-select-rate',
                        required=False,
                        default=0.0,
                        type=ProgramArgumentsConfiguration.percentage_float,
                        help=ProgramArgumentsConfiguration.RANDOM_SELECT_RATE_DESCRIPTION)

    parser.add_argument('--gtsrb-image-path',
                        default='../GTSRB/Final_Training/Images',
                        required=False,
                        type=ProgramArgumentsConfiguration.gtsrb_path,
                        help=ProgramArgumentsConfiguration.RANDOM_SELECT_RATE_DESCRIPTION)


    population_generator_group = parser.add_argument_group('Population Generator',
                                                           ProgramArgumentsConfiguration.POPULATION_GENERATOR_DESCRIPTION)
    group = population_generator_group.add_mutually_exclusive_group(required=True)
    group.add_argument('--rand', action='store_true')
    group.add_argument('--color', action='store_true')
    group.add_argument('--sample', action='store_true')
    group.add_argument('--genetic', action='store_true')
    group.add_argument('--brute-force', action='store_true')
    group.add_argument('--circle', action='store_true')
    group.add_argument('--polygon', action='store_true')
    group.add_argument('--gilogo', action='store_true')
    group.add_argument('--tiles', action='store_true')

    args = parser.parse_args()

    classifier = OnlineClassifier()
    class_id, class_name = ProgramArgumentsConfiguration.get_class(args.target)


    mutation_function = None #Will be set if necessary

    image_path = args.gtsrb_image_path
    size = args.population_size

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
        mutation_function=GeometricMutations.mutate_circle_function()
    elif args.polygon:
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=20,
                                                          population_generator=PolygonPopulationGenerator(100),
                                                          algorithm=GeometricGeneticAlgorithm,
                                                          mutation_intensity=0.05,
                                                          mutation_function=GeometricMutations.mutate_polygon_function(
                                                              n=3))
        mutation_function=GeometricMutations.mutate_polygon_function(n=3)
    elif args.gilogo:
        image = Image.open("gi-logo.jpg")
        inverted_image = PIL.ImageOps.invert(image)
        image = inverted_image.convert("1")
        square_size = 5
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=20,
                                                          population_generator=BitmapPopulationGenerator(100,
                                                                                                         image,
                                                                                                         num_horizontal=square_size,
                                                                                                         num_vertical=square_size),
                                                          algorithm=GeometricGeneticAlgorithm,
                                                          mutation_intensity=0.05,
                                                          mutation_function=GeometricMutations.mutate_bitmap_function(
                                                              img=image, num_horizontal=square_size,
                                                              num_vertical=square_size))

        mutation_function=GeometricMutations.mutate_bitmap_function(img=image)

    elif args.tiles:
        #TODO: DIESE COLORS RANDOM
        color1 = (255, 224, 130)
        color2 = (255, 160, 0)

        mutation_function = GeometricMutations.mutate_tile_function(color1, color2)
        population_generator = GeneticPopulationGenerator(size=size,
                                                          class_id=class_id,
                                                          steps=20,
                                                          population_generator=TilePopulationGenerator(100,
                                                                                                       color1=color1,
                                                                                                       color2=color2),
                                                          algorithm=GeometricGeneticAlgorithm,
                                                          mutation_intensity=0.05,
                                                          mutation_function=mutation_function)
    else:
        population_generator = PopulationGenerator(size=size)

    print()

    if args.circle or args.polygon or args.gilogo or args.tiles:
        genetic = GeometricGeneticAlgorithm(classifier=classifier,
                                            class_to_optimize=class_name,
                                            retain_rate=args.retain_rate,
                                            mutation_rate=args.mutation_rate,
                                            mutation_intensity=args.mutation_intensity,
                                            random_select_rate=args.random_select_rate,
                                            mutation_function=mutation_function
                                            )
    else:
        genetic = GeneticAlgorithm(classifier=classifier,
                                   class_to_optimize=class_name,
                                   retain_rate=args.retain_rate,
                                   mutation_rate=args.mutation_rate,
                                   mutation_intensity=args.mutation_intensity,
                                   random_select_rate=args.random_select_rate)

    population, _ = genetic.run(initial_population_generator=population_generator,
                                grade_limit=args.confidence,
                                steps=100)

    best = max(population, key=lambda individual: individual.classification.value_for_class(class_name=class_name))
    best.image.save("tmp/best_" + str(class_id) + '.png')
