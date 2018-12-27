import random as rd
import string

from PIL import Image

from classifier.classifier import Classifier
from utils.temp_image_file_utilities import TempImageFileUtilities


class ImageUtilities:

    @staticmethod
    def save_image_to_tempfile(image: Image, subdirectory : string = "", file_name : string = None):
        """
        Saves the given image into a temporary file folder.
        :param image: The image which should be saved.
        :param subdirectory: An optional subdirectory in which the file should be stored.
        :param file_name: An optional file name without extension. If no name is provided a random name is generated.
        :return: The complete file path to the newly generated image.
        """
        full_file_path = TempImageFileUtilities.generate_file_name_with_directory(subdirectory = subdirectory, file_name = file_name)
        image.save(full_file_path)
        return full_file_path

    @staticmethod
    def get_empty_image():
        return Image.new('RGB', (Classifier.DESIRED_IMAGE_WIDTH, Classifier.DESIRED_IMAGE_HEIGHT)), \
               Classifier.DESIRED_IMAGE_WIDTH*Classifier.DESIRED_IMAGE_HEIGHT

    @staticmethod
    def mutate_pixel(image : Image, pixel, min = 0, max = 255):
        data = list(image.getdata())
        data[pixel] = (rd.randint(min,max), rd.randint(min,max), rd.randint(min,max))
        image.putdata(data)

    @staticmethod
    def rearrange_image(image : Image):
        data = list(image.getdata())
        rd.shuffle(data)
        image.putdata(data)

    @staticmethod
    def add_image_avg(image1: Image, image2: Image):
        data1 = list(image1.getdata())
        data2 = list(image2.getdata())

        new_image_data = list()

        img, pixel_count = ImageUtilities.get_empty_image()

        for i in range(pixel_count):
            new_image_data.append(tuple(map(lambda x, y: int((x + y) * 0.5), data1[i], data2[i])))

        img.putdata(new_image_data)
        return img

    @staticmethod
    def combine_images(image1 : Image, image2 : Image):
        data1 = list(image1.getdata())
        data2 = list(image2.getdata())

        img, pixel_count = ImageUtilities.get_empty_image()

        pixel_data = list()

        for i in range(pixel_count):
            pixel_data.append(rd.choice((data1[i], data2[i])))

        img.putdata(pixel_data)
        return img
