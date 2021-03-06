import argparse
import pathlib
import time

import PIL.ImageOps
import pandas as pd
from PIL import Image

from classifier.online_classifier import OnlineClassifier
from config.classifier_configuration import ClassifierConfiguration
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.geometric.geometric_mutations import GeometricMutations
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.population_generator.bitmap_population_generator import BitmapPopulationGenerator
from genetic.population_generator.circle_population_generator import CirclePopulationGenerator
from genetic.population_generator.existing_population_generator import ExistingPopulationGenerator
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from genetic.population_generator.polygon_population_generator import PolygonPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import \
    SampleImagesRearrangePopulationGenerator
from genetic.population_generator.single_image_population_generator import SingleImagePopulationGenerator
from genetic.population_generator.tile_population_generator import TilePopulationGenerator
from utils.image_utilities import ImageUtilities
from utils.road_sign_class_mapper_utilities import RoadSignClassMapperUtilities

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confidence', required=True, type=float)
    parser.add_argument('-s', '--steps', type=int, default=100)
    parser.add_argument('-ps', '--pre-steps', type=int, default=20)
    parser.add_argument('-id', '--image-dir', type=str, default='../GTSRB/Final_Training/Images')
    parser.add_argument('-sz', '--size', type=int, default=20)
    parser.add_argument('-mi', '--mutation-intensity', type=float, default=0.1)
    parser.add_argument('-f', '--config', type=str, required=True)
    args = parser.parse_args()

    classifier = OnlineClassifier()
    grade_limit = args.confidence
    image_path = args.image_dir
    size = args.size
    genetic_size = 100
    data = []
    mutation_rate = 1.0
    mutation_intensity = args.mutation_intensity

    label = None
    made_dir = False

    pathlib.Path('tmp/final/').mkdir(parents=True, exist_ok=True)

    config = []
    with open(args.config, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            class_id, method, dir = [token.strip() for token in line.split(',')]
            config.append((int(class_id), method, dir))

    for class_id, method, dir in config:
        print('Class {}'.format(class_id))
        class_name = RoadSignClassMapperUtilities.get_name_by_class(class_id)
        if class_name is None:
            continue

        population_generator = None
        genetic = None

        if method == 'normal':
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=SampleImagesRearrangePopulationGenerator(
                                                                  size=genetic_size, target_class=class_id,
                                                                  image_dir=image_path),
                                                              mutation_intensity=mutation_intensity)
            genetic = GeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                       mutation_intensity=mutation_intensity)

        elif method == 'circle':
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=CirclePopulationGenerator(
                                                                  genetic_size),
                                                              algorithm=GeometricGeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity,
                                                              mutation_function=GeometricMutations.mutate_circle_function())
            genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                                mutation_intensity=mutation_intensity,
                                                mutation_function=GeometricMutations.mutate_circle_function())
        elif method == 'polygon':
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=PolygonPopulationGenerator(
                                                                  genetic_size),
                                                              algorithm=GeometricGeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity,
                                                              mutation_function=GeometricMutations.mutate_polygon_function(
                                                                  n=3))
            genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                                mutation_intensity=mutation_intensity,
                                                mutation_function=GeometricMutations.mutate_polygon_function(
                                                    n=3))
        elif method == 'gilogo':
            image = Image.open("gi-logo.jpg")
            inverted_image = PIL.ImageOps.invert(image)
            image = inverted_image.convert("1")
            square_size = 5
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=BitmapPopulationGenerator(
                                                                  genetic_size, image, num_vertical=square_size,
                                                                  num_horizontal=square_size
                                                              ),
                                                              algorithm=GeometricGeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity,
                                                              mutation_function=GeometricMutations.mutate_bitmap_function(
                                                                  img=image, num_horizontal=square_size,
                                                                  num_vertical=square_size))
            genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                                mutation_intensity=mutation_intensity,
                                                mutation_function=GeometricMutations.mutate_bitmap_function(img=image))
        elif method == 'apple':
            image = Image.open("apple.jpg")
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=SingleImagePopulationGenerator(
                                                                  genetic_size, image),
                                                              algorithm=GeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity,
                                                              pixel_mutation_function=ImageUtilities.mutate_non_dark_pixels)

            if dir:
                population_generator = ExistingPopulationGenerator(size=size, dir=dir)

            genetic = GeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                       mutation_intensity=mutation_intensity,
                                       pixel_mutation_function=ImageUtilities.mutate_non_dark_pixels)
            label = 'apple'

        elif method == 'gilogo2':
            image = Image.open("gi-logo.jpg")
            inverted_image = PIL.ImageOps.invert(image)
            image = inverted_image

            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=SingleImagePopulationGenerator(
                                                                  genetic_size, image),
                                                              algorithm=GeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity,
                                                              pixel_mutation_function=ImageUtilities.mutate_non_dark_pixels)

            if dir:
                population_generator = ExistingPopulationGenerator(size=size, dir=dir)

            genetic = GeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                       mutation_intensity=mutation_intensity,
                                       pixel_mutation_function=ImageUtilities.mutate_non_dark_pixels)
            label = 'gilogo2'

        elif method == 'tiles':
            color1 = (255, 82, 82)
            color2 = (255, 255, 255)
            population_generator = GeneticPopulationGenerator(size=size,
                                                              class_id=class_id,
                                                              steps=args.pre_steps,
                                                              population_generator=TilePopulationGenerator(genetic_size,
                                                                                                           color1=color1,
                                                                                                           color2=color2,
                                                                                                           interpolate=False),
                                                              algorithm=GeometricGeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity,
                                                              mutation_function=GeometricMutations.mutate_tile_function(
                                                                  color1, color2, interpolation=False))
            if dir:
                population_generator = ExistingPopulationGenerator(size=size, dir=dir)

            genetic = GeometricGeneticAlgorithm(classifier=classifier,
                                                class_to_optimize=class_name,
                                                mutation_intensity=mutation_intensity,
                                                mutation_function=GeometricMutations.mutate_tile_function(color1,
                                                                                                          color2,
                                                                                                          interpolation=False))

        start = time.time()

        population, steps = genetic.run(initial_population_generator=population_generator, grade_limit=grade_limit,
                                        steps=args.steps)
        end = time.time()

        best = max(population, key=lambda x: x.classification.value_for_class(class_name))

        best.image.save(
            'tmp/final/{}_{}.{}'.format(class_id, method, ClassifierConfiguration.DESIRED_IMAGE_EXTENSION))

        data.append({
            'class_id': class_id,
            'class_name': class_name,
            'steps': steps,
            'confidence': best.classification.value_for_class(class_name),
            'time': end - start,
            'size': size,
            'mutation_rate': mutation_rate,
            'mutation_intensity': mutation_intensity,
            'grade_limit': grade_limit
        })
        pd.DataFrame(data).to_csv('tmp/final/results.csv')
