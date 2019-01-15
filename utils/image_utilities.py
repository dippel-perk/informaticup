import random as rd
import string

import numpy as np
from PIL import Image
from typing import List

from config.classifier_configuration import ClassifierConfiguration
from utils.temp_image_file_utilities import TempImageFileUtilities


class ImageUtilities:
    """
    Provides basic image realted operations.
    """

    @staticmethod
    def save_image_to_tempfile(image: Image, subdirectory: string = "", file_name: string = None) -> str:
        """
        Saves the given image into a temporary file folder.
        :param image: The image which should be saved.
        :param subdirectory: An optional subdirectory in which the file should be stored.
        :param file_name: An optional file name without extension. If no name is provided a random name is generated.
        :return: The complete file path to the newly generated image.
        """
        full_file_path = TempImageFileUtilities.generate_file_name_with_directory(subdirectory=subdirectory,
                                                                                  file_name=file_name)
        image.save(full_file_path)
        return full_file_path

    @staticmethod
    def get_empty_image() -> Image:
        """
        Returns an empty image with the desired dimensions.
        :return: The image.
        """
        return Image.new('RGB',
                         (ClassifierConfiguration.DESIRED_IMAGE_WIDTH, ClassifierConfiguration.DESIRED_IMAGE_HEIGHT)), \
               ClassifierConfiguration.DESIRED_IMAGE_WIDTH * ClassifierConfiguration.DESIRED_IMAGE_HEIGHT

    @staticmethod
    def mutate_pixel(image: Image.Image, pixel : int, min: int = 0, max: int = 255) -> None:
        """
        Mutates a given pixel and sets it to a random pixel values. Uses passed variables to specify what color range
        the pixel can have.
        :param image: The image which should be mutated.
        :param pixel: The pixel index.
        :param min: The minimum r,g and b value
        :param max: The maximum r,g and b value
        :return: None
        """

        assert min <= max
        assert min >= 0 and max <= 255

        indices = np.unravel_index(pixel, ClassifierConfiguration.DESIRED_IMAGE_DIMENSIONS)
        image.putpixel((indices[1], indices[0]), (rd.randint(min, max), rd.randint(min, max), rd.randint(min, max)))

    @staticmethod
    def mutate_pixels(image: Image, pixels: List[int], min: int = 0, max: int = 255):
        """
        Mutates the given pixels and sets it to a random pixel values. Uses passed variables to specify what color range
        the pixel can have.
        :param image: The image which should be mutated.
        :param pixels: The pixel indexes which should be mutated.
        :param min: The minimum r,g and b value
        :param max: The maximum r,g and b value
        :return: None
        """
        for pixel in pixels:
            ImageUtilities.mutate_pixel(image, pixel, min, max)

    @staticmethod
    def mutate_non_dark_pixels(image: Image, pixels: List[int], min: int = 0, max: int = 255):
        """
        Mutates the given pixels and sets it to a random pixel values. Uses passed variables to specify what color range
        the pixel can have. Black pixel get ignored.
        :param image: The image which should be mutated.
        :param pixels: The pixel indexes which should be mutated.
        :param min: The minimum r,g and b value
        :param max: The maximum r,g and b value
        :return: None
        """
        data = image.getdata()
        for pixel in pixels:
            if data[pixel] != (0, 0, 0):
                ImageUtilities.mutate_pixel(image, pixel, min, max)

    @staticmethod
    def rearrange_image(image: Image):
        """
        Given an image, rearranges the pixels of the image and returns the resulting permutated image.
        :param image: The input image.
        :return: The permutated image.
        """
        data = list(image.getdata())
        rd.shuffle(data)
        image.putdata(data)

    @staticmethod
    def combine_images(image1: Image, image2: Image):
        """
        Given two images, combine both images so that for each pixel of the result image either the pixel value from
        image1 oder the pixel value from image2 gets applied.
        :param image1: The first image.
        :param image2: The second image
        :return: The combined image.
        """
        data1 = image1.getdata()
        data2 = image2.getdata()

        img, pixel_count = ImageUtilities.get_empty_image()

        pixel_data = [rd.choice([pixel_1, pixel_2]) for pixel_1, pixel_2 in zip(data1, data2)]

        img.putdata(pixel_data)
        return img
