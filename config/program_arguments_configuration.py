import argparse
import os
from road_sign_class_mapper import RoadSignClassMapper

class ProgramArgumentsConfiguration:

    PROGRAM_DESCRIPTION = "This tool perfroms a black box attack against a pretrained neural network as part of the informaticup competition. " \
                          "It generates adversarial images and saves them to a given output directory."

    TARGET_CLASS_DESCRIPTION = "The target class name or target class id."
    CONFIDENCE_DESCRIPTION = "The desired confidence. The algorithm will terminate, when the confidence is reached by at least one generated image. " \
                             "If no confidence is provided by the user, the algorithm will run the maximum number of steps."
    STEPS_DESCRIPTION = "The maximum number of steps the algorithm will run."
    MUTATION_RATE_DESCRIPTION = "The mutation rate which should be used."
    MUTATION_INTENSITY_DESCRIPTION = "The mutation intensity which should be used."
    POPULATION_SIZE_DESCRIPTION = "The desired population size."
    RANDOM_SELECT_RATE_DESCRIPTION = "The percentage of individuals which should be randomly added to the next generation."

    GTSRB_IMAGE_PATH_DESCRIPTION = "The percentage of individuals which should be randomly added to the next generation."
    POPULATION_GENERATOR_DESCRIPTION = "Determines the population generator, which should be used. The " \
                                        "population generator might have influence on the mutation and crossover function."
    @staticmethod
    def percentage_float(x):
        """
        Validates a given input number x so that it is between 0 and 1.
        :param x: The input number.
        :return: The validated number in range [0,1]
        """
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"% (x))
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
                class_name = RoadSignClassMapper().get_name_by_class(class_id=class_id)
            else:
                class_name = x
                class_id = RoadSignClassMapper().get_class_by_name(name=class_name)
        except ValueError:
            #This error occurs when a class could not be found
            raise argparse.ArgumentTypeError("The provided class name or id was not found")


        if class_id is None or class_name is None:
            raise argparse.ArgumentTypeError("The provided class name or id can not be recognized by the online neural network. Please try another one.")

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
            raise argparse.ArgumentTypeError("%r is not a valid directory"%(x,))
        return x

    @staticmethod
    def get_class(x : str):
        """
        For a given string x this method extracts the class name and class id.
        :param x: The input string.
        :return: A tuple containing the class id and the class name.
        """

        if not ProgramArgumentsConfiguration.target_class(x):
            return

        if x.isdigit():
            class_id = int(x)
            class_name = RoadSignClassMapper().get_name_by_class(class_id=class_id)
        else:
            class_name = x
            class_id = RoadSignClassMapper().get_class_by_name(name=class_name)

        return (class_id, class_name)