import argparse

from genetic.basic_approach import BasicApproach
from genetic.population_generator.sample_images_rearrange_population_generator import \
    SampleImagesRearrangePopulationGenerator
from genetic.population_generator.genetic_population_generator import GeneticPopulationGenerator
from classifier.online_classifier import OnlineClassifier
from road_sign_class_mapper import RoadSignClassMapper
from classifier.classifier import Classifier
import pandas as pd

if __name__ == '__main__':
    classifier = OnlineClassifier()
    grade_limit = 0.9
    image_path = '../GTSRB/Final_Training/Images'
    size = 20
    data = []
    mutation_rate = 0.05

    for class_id in range(43):
        print('Class {}'.format(class_id))
        class_name = RoadSignClassMapper().get_name_by_class(class_id)
        if class_name is None:
            continue

        genetic = BasicApproach(classifier=classifier, class_to_optimize=class_name, mutation_rate=mutation_rate)

        population_generator = GeneticPopulationGenerator(size=size, class_id=class_id, steps=20,
                                                          population_generator=SampleImagesRearrangePopulationGenerator(
                                                              size=100, target_class=class_id,
                                                              image_dir=image_path))

        population, steps = genetic.run(initial_population_generator=population_generator, grade_limit=grade_limit,
                                        steps=100)

        best = max(population, key=lambda x: x.classification.value_for_class(class_name))

        best.image.save('tmp/best/{}.{}'.format(class_id, Classifier.DESIRED_IMAGE_EXTENSION))

        data.append({
            'class_id': class_id,
            'class_name': class_name,
            'steps': steps,
            'confidence': best.classification.value_for_class(class_name),
            'size': size,
            'mutation_rate': mutation_rate,
            'grade_limit': grade_limit
        })
        pd.DataFrame(data).to_csv('results.csv')

    print(OnlineClassifier.SEEN_CLASSES)