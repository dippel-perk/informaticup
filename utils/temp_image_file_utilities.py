import os.path
import pathlib as pathlib
import string

import numpy as np

from config.classifier_configuration import ClassifierConfiguration


class TempImageFileUtilities:
    TEMP_DIR = "./tmp/generated_images/"
    FILE_NAME_LENGTH = 16

    @staticmethod
    def generate_file_name_with_directory(subdirectory: string = "", file_name: string = None) -> str:
        """
        Generates a relative file path while ensuring that all directories of the path exist
        :param subdirectory: An optional subdirectory in which the file should be stored.
        :param file_name: An optional file name without extension. If no name is provided a random name is generated.
        :return: A complete file path to the file. The file does not exist yet.
        """
        full_directory_path = os.path.join(TempImageFileUtilities.TEMP_DIR, subdirectory)
        pathlib.Path(full_directory_path).mkdir(parents=True, exist_ok=True)

        if file_name:
            file_name = TempImageFileUtilities.__add_prefix_to_file_name(file_name)
        else:
            file_name = TempImageFileUtilities.__generate_random_file_name()

        complete_file_path = os.path.join(full_directory_path, file_name)
        assert not os.path.isfile(complete_file_path)
        return complete_file_path

    @staticmethod
    def __generate_random_file_name() -> str:
        """
        Generates a random file name
        :return: The file name.
        """
        return TempImageFileUtilities.__add_prefix_to_file_name(
            ''.join(np.random.choice(list(string.ascii_letters + string.digits),
                                     size=(TempImageFileUtilities.FILE_NAME_LENGTH))
                    )
        )

    @staticmethod
    def __add_prefix_to_file_name(file_name: string) -> str:
        """
        IAdds the desired image extension to a given file name.
        :param file_name: The file name.
        :return: The file name with extension.
        """
        assert not file_name.endswith("." + ClassifierConfiguration.DESIRED_IMAGE_EXTENSION)
        return file_name + "." + ClassifierConfiguration.DESIRED_IMAGE_EXTENSION
