from genetic.basic_approach import BasicApproach
from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from api import NeuralNetworkAPI
if __name__ == '__main__':


    genetic = BasicApproach(class_to_optimize="Vorfahrt")

    api = NeuralNetworkAPI()

    population_generator = RandomPopulationGenerator(api=api, size=10)
    genetic.run(initial_population_generator=population_generator, steps=10)