import argparse
import os

from utils.road_sign_class_mapper_utilities import RoadSignClassMapperUtilities


class ProgramArgumentsConfiguration:
    PROGRAM_DESCRIPTION = "This tool perfroms a black box attack against a pretrained neural network as part of " \
                          "the informaticup competition. It generates adversarial images and saves them to a given " \
                          "output directory."

    TARGET_CLASS_DESCRIPTION = "The target class name or target class id."
    CONFIDENCE_DESCRIPTION = "The desired confidence. The algorithm will terminate, when the confidence is reached " \
                             "by at least one generated image. If no confidence is provided by the user, the algorithm " \
                             "will run the maximum number of steps. The confidence must be in range [0,1]."
    STEPS_DESCRIPTION = "The maximum number of steps, i.e. the number of the generations the algorithm will evolve."
    MUTATION_RATE_DESCRIPTION = "The mutation rate which should be used."
    MUTATION_INTENSITY_DESCRIPTION = "The mutation intensity which should be used."
    POPULATION_SIZE_DESCRIPTION = "The desired population size. Every generation of the genetic algorithm will have this many individuals."
    RANDOM_SELECT_RATE_DESCRIPTION = "The percentage of individuals which should be randomly added to the next generation."

    GTSRB_IMAGE_PATH_DESCRIPTION = "The path to the German Traffic Sign Recognition Benchmark. The images of the " \
                                   "benchmark are used in some population generators."

    IMAGE_PATH_DESCRIPTION = "A path to a jpg or jpeg image. The file will be used for the grid image or single image " \
                             "population generators. The background must be black and the foreground must be white. "

    POPULATION_GENERATOR_DESCRIPTION = "Determines the population generator, which should be used. The population " \
                                       "generator might have influence on the mutation and crossover function. " \
                                       "At least one population generator has to be chosen from the set of possibilities."

    OUT_PATH_DESCRIPTION = "A complete history of all computed generations will be saved to an output directory. " \
                           "If no directory is provided, a standard directory is chosen."

    SUBSTITUTE_DESCRIPTION = "If the substitute command was added, the selected population generator will be the initial " \
                             "population generator for the substitute network. The tool will evolve this population to " \
                             "a state where the substitute network classifies the individuals to be most likely in the " \
                             "target class. The substitute has to be added **after the other arguments. If added, the " \
                             "user has two additional arguments."

    SUBSTITUTE_POPULATION_SIZE_DESCRIPTION = "The population size which should be used by the substitute network. " \
                                             "The size has to be larger than --population-size n, because we want to " \
                                             "use the most fit n individuals of the substitute network as an initial " \
                                             "population of the genetic algorithm. If the variable is not set, we use " \
                                             "the size 3n."
    SUBSTITUTE_STEPS_DESCRIPTION = "Determines the amount of steps which should be performed on the substitute network."

    RAND_POPULATION_GENERATOR_DESCRIPTION = "Generates a population of random image individuals."
    COLOR_POPULATION_GENERATOR_DESCRIPTION = "Generates a population with the same color distribution as some random training images"
    SAMPLE_POPULATION_GENERATOR_DESCRIPTION = "Generates a population, which contains rearrangements of training set images."
    BRUTE_FORCE_POPULATION_GENERATOR_DESCRIPTION = "Generates a population of random image individuals, while ensuring that each individual's classification contains the target class. To achieve this, new images are generated until the target class is part of the classification."
    CIRCLE_POPULATION_GENERATOR_DESCRIPTION = "Generates population of geometric individuals, which are filled with random circles."
    POLYGON_POPULATION_GENERATOR_DESCRIPTION = "Generates a population of geometric individuals which contain random polygons. We restricted the polygons to be triangles."
    TILE_POPULATION_GENERATOR_DESCRIPTION = "Generates a population with geometric individuals. Every individual is completely filled with so called tiles."

    GRID_IMAGE_POPULATION_GENERATOR_DESCRIPTION = "Generates a population of the given image on a grid with different color."
    SINGLE_IMAGE_POPULATION_GENERATOR_DESCRIPTION = "Generates a population of the given image filled with random pixels."

    @staticmethod
    def percentage_float(x):
        """
        Validates a given input number x so that it is between 0 and 1.
        :param x: The input number.
        :return: The validated number in range [0,1]
        """
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x))
        return x

    @staticmethod
    def target_class(x):
        """
        Validates that a given class id or class name is valid.
        :param x: The input string.
        :return: A validated class id or class name.
        """
        try:
            if x.isdigit():
                class_id = int(x)
                class_name = RoadSignClassMapperUtilities.get_name_by_class(class_id=class_id)
            else:
                class_name = x
                class_id = RoadSignClassMapperUtilities.get_class_by_name(name=class_name)
        except ValueError:
            # This error occurs when a class could not be found
            raise argparse.ArgumentTypeError("The provided class name or id was not found")

        if class_id is None or class_name is None:
            raise argparse.ArgumentTypeError(
                "The provided class name or id can not be recognized by the online neural network. Please try another one.")

        return x

    @staticmethod
    def gtsrb_path(x):
        """
        Validates that a certain path is a valid path.
        :param x: The input path.
        :return: The validated output path.
        """
        x = str(x)
        if not os.path.isdir(x):
            raise argparse.ArgumentTypeError("%r is not a valid directory" % (x,))
        return x

    @staticmethod
    def image_file(x):
        """
        Validates that a certain file is a valid image file.
        :param x: The input file.
        :return: The validated image path.
        """
        if x is None:
            return x

        x = str(x)
        if not os.path.isfile(x):
            raise argparse.ArgumentTypeError("%r is not a file" % (x,))
        if not x.endswith(".jpg") and not x.endswith(".jpeg"):
            raise argparse.ArgumentTypeError("%r is not a jpg or jpeg file" % (x,))
        return x
