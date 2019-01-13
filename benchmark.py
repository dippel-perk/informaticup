import argparse
import pathlib
import time

import PIL.ImageOps
import pandas as pd
from PIL import Image
from genetic.population_generator.geometric.bitmap_population_generator import BitmapPopulationGenerator
from genetic.population_generator.geometric.polygon_population_generator import PolygonPopulationGenerator

from classifier.classifier import Classifier
from classifier.online_classifier import OnlineClassifier
from genetic.genetic_algorithm import GeneticAlgorithm
from genetic.geometric.geometric_mutations import GeometricMutations
from genetic.geometric_genetic_algorithm import GeometricGeneticAlgorithm
from genetic.population_generator.circle_population_generator import CirclePopulationGenerator
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from genetic.population_generator.sample_images_rearrange_population_generator import SampleImagesRearrangePopulationGenerator
from road_sign_class_mapper import RoadSignClassMapper

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confidence', required=True, type=float)
    parser.add_argument('-s', '--steps', type=float, default=100)
    parser.add_argument('-ps', '--pre-steps', type=float, default=20)
    parser.add_argument('-id', '--image-dir', type=str, default='../GTSRB/Final_Training/Images')
    parser.add_argument('-sz', '--size', type=float, default=20)
    parser.add_argument('-si', '--start-index', type=int, default=0)
    parser.add_argument('-ei', '--end-index', type=int, default=43)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--normal', action='store_true')
    group.add_argument('--circle', action='store_true')
    group.add_argument('--polygon', action='store_true')
    group.add_argument('--gilogo', action='store_true')

    args = parser.parse_args()

    classifier = OnlineClassifier()
    grade_limit = args.confidence
    image_path = args.image_dir
    size = args.size
    genetic_size = 100
    data = []
    mutation_rate = 1.0
    mutation_intensity = 0.1

    label = None
    made_dir = False

    for class_id in range(args.start_index, args.end_index):
        print('Class {}'.format(class_id))
        class_name = RoadSignClassMapper().get_name_by_class(class_id)
        if class_name is None:
            continue

        population_generator = None
        genetic = None

        if args.normal:
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=SampleImagesRearrangePopulationGenerator(
                                                                  size=genetic_size, target_class=class_id,
                                                                  image_dir=image_path),
                                                              mutation_intensity=mutation_intensity)
            genetic = GeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                    mutation_intensity=mutation_intensity)
            label = "normal"

        elif args.circle:
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=CirclePopulationGenerator(
                                                                  genetic_size),
                                                              algorithm=GeometricGeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity)
            genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                                mutation_intensity=mutation_intensity,
                                                mutation_function=GeometricMutations.mutate_circle_function())
            label = "circle"
        elif args.polygon:
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=PolygonPopulationGenerator(
                                                                  genetic_size),
                                                              algorithm=GeometricGeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity)
            genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                                mutation_intensity=mutation_intensity,
                                                mutation_function=GeometricMutations.mutate_polygon_function(
                                                    n=3))
            label = "polygon"
        elif args.gilogo:
            image = Image.open("gi-logo.jpg")
            inverted_image = PIL.ImageOps.invert(image)
            image = inverted_image.convert("1").resize((200, 200))
            population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=args.pre_steps,
                                                              population_generator=BitmapPopulationGenerator(
                                                                  genetic_size, image,
                                                                  avg_num=50),
                                                              algorithm=GeometricGeneticAlgorithm,
                                                              mutation_intensity=mutation_intensity)
            genetic = GeometricGeneticAlgorithm(classifier=classifier, class_to_optimize=class_name,
                                                mutation_intensity=mutation_intensity,
                                                mutation_function=GeometricMutations.mutate_bitmap_function(img=image))
            label = "gilogo"

        start = time.time()

        population, steps = genetic.run(initial_population_generator=population_generator, grade_limit=grade_limit,
                                        steps=args.steps)
        end = time.time()

        best = max(population, key=lambda x: x.classification.value_for_class(class_name))

        if not made_dir:
            pathlib.Path('tmp/best/{}/'.format(label)).mkdir(parents=True, exist_ok=True)
            made_dir = True

        best.image.save('tmp/best/{}/{}.{}'.format(label, class_id, Classifier.DESIRED_IMAGE_EXTENSION))

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
        pd.DataFrame(data).to_csv('results.csv')
