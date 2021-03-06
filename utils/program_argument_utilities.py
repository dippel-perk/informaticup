from PIL import Image, ImageOps

from classifier.classifier import Classifier
from config.program_argument_utilities_configuration import ProgramArgumentUtilitiesConfiguration
from genetic.geometric.geometric_mutations import GeometricMutations
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.grade_average_genetic_algorithm import GradeAverageGeneticAlgorithm
from genetic.population_generator.bitmap_population_generator import BitmapPopulationGenerator
from genetic.population_generator.circle_population_generator import CirclePopulationGenerator
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from genetic.population_generator.polygon_population_generator import PolygonPopulationGenerator
from genetic.population_generator.random_brute_force_population_generator import RandomBruteForcePopulationGenerator
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import \
    SampleImagesRearrangePopulationGenerator
from genetic.population_generator.tile_population_generator import TilePopulationGenerator
from genetic.population_generator.train_color_population_generator import TrainColorPopulationGenerator
from genetic.population_generator.single_image_population_generator import SingleImagePopulationGenerator
from utils.image_utilities import ImageUtilities
from utils.road_sign_class_mapper_utilities import RoadSignClassMapperUtilities
from utils.output_utilities import print_error


class ProgramArgumentUtilities:
    @staticmethod
    def get_class(x: str):
        """
        For a given string x this method extracts the class name and class id.
        :param x: The input string.
        :return: A tuple containing the class id and the class name.
        """

        if x.isdigit():
            class_id = int(x)
            class_name = RoadSignClassMapperUtilities.get_name_by_class(class_id=class_id)
        else:
            class_name = x
            class_id = RoadSignClassMapperUtilities.get_class_by_name(name=class_name)

        return (class_id, class_name)

    @staticmethod
    def get_population_generator_from_args(args, classifier: Classifier):
        """
        Given the args of the argparse parse_args function, this method generates the appropriate population
        generator as well as some important attributes.
        :param args: The argparse result.
        :param classifier: The classifier which should be used.
        :return: The class id, class name, population generator, mutation function and genetic algorithm
        """
        class_id, class_name = ProgramArgumentUtilities.get_class(args.target)

        # Will be set if necessary
        mutation_function = None
        genetic_algorithm = GradeAverageGeneticAlgorithm

        image_path = args.gtsrb_image_path

        if args.population_generator == ProgramArgumentUtilitiesConfiguration.DEFAULT_POPULATION_GENERATOR:
            size = args.population_size
        elif args.population_generator == ProgramArgumentUtilitiesConfiguration.GENETIC_POPULATION_GENERATOR:

            if args.substitute_population_size is None:
                size = args.population_size * 3
            else:
                if args.substitute_population_size < args.population_size:
                    print_error("It makes no sense that the substitute network has a smaller population than the online network.")
                    exit()
                assert args.substitute_population_size >= args.population_size
                size = args.substitute_population_size

        else:
            raise ValueError("Please chose either genetic or non-genetic population generation.")

        if args.color:
            population_generator = TrainColorPopulationGenerator(size=size,
                                                                 target_class=class_id,
                                                                 image_dir=image_path)
            mutation_function = ImageUtilities.mutate_pixels
        elif args.rand:
            population_generator = RandomPopulationGenerator(size=size)
            mutation_function = ImageUtilities.mutate_pixels
        elif args.sample:
            population_generator = SampleImagesRearrangePopulationGenerator(size=size,
                                                                            target_class=class_id,
                                                                            image_dir=image_path)
            mutation_function = ImageUtilities.mutate_pixels
        elif args.brute_force:

            if args.population_generator == ProgramArgumentUtilitiesConfiguration.GENETIC_POPULATION_GENERATOR:
                print_error("It makes no sense to execute the brute force generator (which forces online classification) with the substitute network.")
                exit()
            population_generator = RandomBruteForcePopulationGenerator(size=size,
                                                                       classifier=classifier,
                                                                       target_class=class_name)
            mutation_function = ImageUtilities.mutate_pixels
        elif args.circle:
            population_generator = CirclePopulationGenerator(size=size)
            genetic_algorithm = GeometricGeneticAlgorithm
            mutation_function = GeometricMutations.mutate_circle_function()

        elif args.polygon:
            population_generator = PolygonPopulationGenerator(size=size)
            genetic_algorithm = GeometricGeneticAlgorithm
            mutation_function = GeometricMutations.mutate_polygon_function(n=3)

        elif args.image_grid:

            if args.image is None:
                print_error("You have to provide an --image when using the image-grid option")
                exit()
            image = Image.open(args.image)
            image = image.convert("1")
            square_size = 5
            population_generator = BitmapPopulationGenerator(size=size,
                                                             img=image,
                                                             num_horizontal=square_size,
                                                             num_vertical=square_size)

            genetic_algorithm = GeometricGeneticAlgorithm

            mutation_function = GeometricMutations.mutate_bitmap_function(img=image,
                                                                          num_horizontal=square_size,
                                                                          num_vertical=square_size)

        elif args.single_image:
            if args.image is None:
                print_error("You have to provide an --image when using the single-image option")
                exit()

            image = Image.open(args.image)
            population_generator = SingleImagePopulationGenerator(size=size, img=image)

            mutation_function = ImageUtilities.mutate_non_dark_pixels

        elif args.tiles:

            population_generator = TilePopulationGenerator(size=size,
                                                           interpolate=False)

            genetic_algorithm = GeometricGeneticAlgorithm
            mutation_function = GeometricMutations.mutate_tile_function(interpolation=False)
        else:
            raise ValueError("Please provide a valid population generation method.")

        if args.population_generator == ProgramArgumentUtilitiesConfiguration.GENETIC_POPULATION_GENERATOR:
            population_generator = GeneticPopulationGenerator(size=args.population_size,
                                                              class_id=class_id,
                                                              steps=args.substitute_population_steps,
                                                              population_generator=population_generator,
                                                              algorithm=genetic_algorithm,
                                                              mutation_intensity=args.mutation_intensity,
                                                              mutation_function=mutation_function,
                                                              pixel_mutation_function=mutation_function)

        return (class_id, class_name, population_generator, mutation_function, genetic_algorithm)
