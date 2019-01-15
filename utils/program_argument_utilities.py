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
        genetic_population_generator_mutation_intensity = 0.1
        genetic_algorithm = GradeAverageGeneticAlgorithm

        image_path = args.gtsrb_image_path

        if args.population_generator == ProgramArgumentUtilitiesConfiguration.DEFAULT_POPULATION_GENERATOR:
            size = args.population_size
        elif args.population_generator == ProgramArgumentUtilitiesConfiguration.GENETIC_POPULATION_GENERATOR:

            if args.genetic_population_size is None:
                size = args.population_size * 3
            else:
                size = args.genetic_population_size

        else:
            raise ValueError("Please chose either genetic or non-genetic population generation.")

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
        elif args.brute_force:
            population_generator = RandomBruteForcePopulationGenerator(size=size,
                                                                       classifier=classifier,
                                                                       target_class=class_name)
        elif args.circle:
            population_generator = CirclePopulationGenerator(size=size)
            genetic_algorithm = GeometricGeneticAlgorithm
            mutation_function = GeometricMutations.mutate_circle_function()
            genetic_population_generator_mutation_intensity = 0.05

        elif args.polygon:
            population_generator = PolygonPopulationGenerator(size=size)
            genetic_algorithm = GeometricGeneticAlgorithm
            mutation_function = GeometricMutations.mutate_polygon_function(n=3)
            genetic_population_generator_mutation_intensity = 0.05

        elif args.gilogo:
            image = Image.open("./resources/gi-logo.jpg")
            inverted_image = ImageOps.invert(image)
            image = inverted_image.convert("1")
            square_size = 5
            population_generator = BitmapPopulationGenerator(size=size,
                                                             img=image,
                                                             num_horizontal=square_size,
                                                             num_vertical=square_size)

            genetic_algorithm = GeometricGeneticAlgorithm

            mutation_function = GeometricMutations.mutate_bitmap_function(img=image,
                                                                          num_horizontal=square_size,
                                                                          num_vertical=square_size)
            genetic_population_generator_mutation_intensity = 0.05

        elif args.snowflake:
            image = Image.open("resources/snowflake.jpeg")
            inverted_image = ImageOps.invert(image)
            image = inverted_image.convert("1")
            square_size = 5
            population_generator = BitmapPopulationGenerator(size=size,
                                                             img=image,
                                                             num_horizontal=square_size,
                                                             num_vertical=square_size)

            genetic_algorithm = GeometricGeneticAlgorithm

            mutation_function = GeometricMutations.mutate_bitmap_random_function(img=image,
                                                                          num_horizontal=square_size,
                                                                          num_vertical=square_size)
            genetic_population_generator_mutation_intensity = 0.05

        elif args.batman:
            image = Image.open("resources/gi-logo.jpg")
            population_generator = SingleImagePopulationGenerator(size=size, img=image)


            mutation_function =  ImageUtilities.mutate_non_dark_pixels
            genetic_population_generator_mutation_intensity = 0.05

        elif args.tiles:
            # TODO: DIESE COLORS RANDOM
            color1 = (255, 224, 130)
            color2 = (255, 160, 0)

            population_generator = TilePopulationGenerator(size=size,
                                                           color1=color1,
                                                           color2=color2)

            genetic_algorithm = GeometricGeneticAlgorithm
            mutation_function = GeometricMutations.mutate_tile_function(color1, color2)
            genetic_population_generator_mutation_intensity = 0.05

        else:
            raise ValueError("Please provide a valid population generation method.")

        if args.population_generator == ProgramArgumentUtilitiesConfiguration.GENETIC_POPULATION_GENERATOR:
            population_generator = GeneticPopulationGenerator(size=args.population_size,
                                                              class_id=class_id,
                                                              steps=args.genetic_population_steps,
                                                              population_generator=population_generator,
                                                              algorithm=genetic_algorithm,
                                                              mutation_intensity=genetic_population_generator_mutation_intensity,
                                                              mutation_function=mutation_function)

        return (class_id, class_name, population_generator, mutation_function, genetic_algorithm)
