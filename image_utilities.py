from api import NeuralNetworkAPI
from PIL import Image
import random as rd
import string
import pathlib as pathlib
import os.path

class ImageUtilities:

    TEMP_DIR = "./tmp/generated_images/"
    FILE_NAME_LENGTH = 16

    @staticmethod
    def save_image_to_tempfile(image: Image):
        pathlib.Path(ImageUtilities.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        file = os.path.join(ImageUtilities.TEMP_DIR, ImageUtilities.generate_file_name())
        image.save(file)
        return file

    @staticmethod
    def generate_file_name():
        return ''.join(rd.choices(string.ascii_letters + string.digits, k=ImageUtilities.FILE_NAME_LENGTH)) + "." + NeuralNetworkAPI.DESIRED_IMAGE_EXTENSION

    @staticmethod
    def get_empty_image():
        return Image.new('RGB', (NeuralNetworkAPI.DESIRED_IMAGE_WIDTH, NeuralNetworkAPI.DESIRED_IMAGE_HEIGHT)), \
               NeuralNetworkAPI.DESIRED_IMAGE_WIDTH*NeuralNetworkAPI.DESIRED_IMAGE_HEIGHT

    @staticmethod
    def generate_black_to_white():

        img, pixel_count = ImageUtilities.get_empty_image()

        pixel_data = list()

        for i in range(pixel_count):

            pixel_data.append((int(255*i / pixel_count),int(255*i / pixel_count), int(255*i / pixel_count)))

        img.putdata(pixel_data)

        return img

    @staticmethod
    def generate_random_noise():

        img, pixel_count = ImageUtilities.get_empty_image()

        pixel_data = list()

        for i in range(pixel_count):
            pixel_data.append((rd.randint(0,255),rd.randint(0,255), rd.randint(0,255)))

        img.putdata(pixel_data)

        return img

    @staticmethod
    def mutate_pixel(image : Image, pixel, min = 0, max = 255):
        data = list(image.getdata())
        data[pixel] = (rd.randint(min,max), rd.randint(min,max), rd.randint(min,max))
        image.putdata(data)

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
