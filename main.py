from genetic.basic_approach import BasicApproach
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from genetic.population_generator.train_color_population_generator import TrainColorPopulationGenerator
from classifier.online_classifier import OnlineClassifier
from road_sign_class_mapper import RoadSignClassMapper

if __name__ == '__main__':

    classifier = OnlineClassifier()

    class_name = "Zulässige Höchstgeschwindigkeit (30)"
    class_id = RoadSignClassMapper().get_class_by_name(name=class_name)

    genetic = BasicApproach(classifier=classifier, class_to_optimize=class_name)


    population_generator = TrainColorPopulationGenerator(size=10, target_class=class_id, image_dir='../GTSRB/Final_Training/')
    #population_generator = RandomPopulationGenerator(size=10)


    genetic.run(initial_population_generator=population_generator, steps=0)