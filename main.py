import argparse
import os

from classifier.online_classifier import OnlineClassifier
from config.program_argument_configuration import ProgramArgumentsConfiguration
from config.program_argument_utilities_configuration import ProgramArgumentUtilitiesConfiguration
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from utils.output_utilities import print_error, print_variable, print_info, print_space, make_bold
from utils.program_argument_utilities import ProgramArgumentUtilities

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=ProgramArgumentsConfiguration.PROGRAM_DESCRIPTION)
    parser.set_defaults(population_generator=ProgramArgumentUtilitiesConfiguration.DEFAULT_POPULATION_GENERATOR)

    subparsers = parser.add_subparsers()

    genetic_population_parser = subparsers.add_parser("substitute",
                                                      help=ProgramArgumentsConfiguration.SUBSTITUTE_DESCRIPTION)

    genetic_population_parser.add_argument("-spn", "--substitute-population-size",
                                           type=int,
                                           help=ProgramArgumentsConfiguration.SUBSTITUTE_POPULATION_SIZE_DESCRIPTION)
    genetic_population_parser.add_argument("-sps", "--substitute-population-steps",
                                           type=int,
                                           default=20,
                                           help=ProgramArgumentsConfiguration.SUBSTITUTE_STEPS_DESCRIPTION)

    genetic_population_parser.set_defaults(population_generator=
                                           ProgramArgumentUtilitiesConfiguration.GENETIC_POPULATION_GENERATOR)

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
                        default=50,
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

    parser.add_argument('--image',
                        default='resources/apple.jpg',
                        required=False,
                        type=ProgramArgumentsConfiguration.image_file,
                        help=ProgramArgumentsConfiguration.IMAGE_PATH_DESCRIPTION)

    parser.add_argument('-o', '--out',
                        required=False,
                        type=str,
                        help=ProgramArgumentsConfiguration.OUT_PATH_DESCRIPTION)

    population_generator_group = parser.add_argument_group('Population Generator',
                                                           ProgramArgumentsConfiguration.POPULATION_GENERATOR_DESCRIPTION)
    group = population_generator_group.add_mutually_exclusive_group(required=True)
    group.add_argument('--rand',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.RAND_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--color',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.COLOR_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--sample',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.SAMPLE_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--brute-force',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.BRUTE_FORCE_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--circle',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.CIRCLE_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--polygon',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.POLYGON_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--image-grid',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.GRID_IMAGE_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--single-image',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.SINGLE_IMAGE_POPULATION_GENERATOR_DESCRIPTION)
    group.add_argument('--tiles',
                       action='store_true',
                       help=ProgramArgumentsConfiguration.TILE_POPULATION_GENERATOR_DESCRIPTION)



    args = parser.parse_args()

    print_info("Initializing the genetic algorithm")

    classifier = OnlineClassifier()
    class_id, class_name, population_generator, mutation_function, algorithm = \
        ProgramArgumentUtilities.get_population_generator_from_args(args=args, classifier=classifier)

    if algorithm == GeometricGeneticAlgorithm:
        genetic = GeometricGeneticAlgorithm(classifier=classifier,
                                            class_to_optimize=class_name,
                                            retain_rate=args.retain_rate,
                                            mutation_rate=args.mutation_rate,
                                            mutation_intensity=args.mutation_intensity,
                                            random_select_rate=args.random_select_rate,
                                            mutation_function=mutation_function,
                                            output_dir=args.out
                                            )
    else:
        genetic = GeneticAlgorithm(classifier=classifier,
                                   class_to_optimize=class_name,
                                   retain_rate=args.retain_rate,
                                   mutation_rate=args.mutation_rate,
                                   mutation_intensity=args.mutation_intensity,
                                   random_select_rate=args.random_select_rate,
                                   pixel_mutation_function=mutation_function,
                                   output_dir=args.out)

    try:
        print_info("The genetic algorithm will be executed with the following configurations")
        print_variable("Algorithm", make_bold(genetic.__repr__()))
        print_variable("Output Directory", "Standard" if args.out is None else args.out)
        print_variable("Class ID", str(class_id))
        print_variable("Class Name", class_name)
        print_variable("Population Size", str(population_generator.size))
        print_variable("Population Generator", str(population_generator))
        print_variable("Stop with Confidence", "Unlimited" if args.confidence > 1.0 else str(args.confidence))
        print_variable("Maximum Steps", str(args.steps))
        print_variable("Retain Rate", str(args.retain_rate))
        print_variable("Mutation Rate", str(args.mutation_rate))
        print_variable("Mutation Intensity", str(args.mutation_intensity))
        print_variable("Random Select Rate", str(args.random_select_rate))
        print_variable("Image File (if necessary)", str(args.image))

        print_space()

        population, _ = genetic.run(initial_population_generator=population_generator,
                                    grade_limit=args.confidence,
                                    steps=args.steps)

        best = max(population, key=lambda individual: individual.classification.value_for_class(class_name=class_name))
        best_file_name = "best_" + str(class_id) + '.png'
        best_complete_path = os.path.join(genetic.output_dir, best_file_name)
        best.image.save(best_complete_path)

        print_info("Saved the best individual with confidence %f%% to %s" % (best.classification.value_for_class(class_name=class_name)*100, best_complete_path))

    except KeyboardInterrupt:
        print_error("\nComputation terminated by user")
