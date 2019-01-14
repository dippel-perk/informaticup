from genetic.population_generator.population_generator import PopulationGenerator
from genetic.image_individual import ImageIndividual
from PIL import Image
import os


class ExistingPopulationGenerator(PopulationGenerator):

    def __init__(self, size: int, dir):
        super().__init__(size=size)
        self.dir = dir

    def __iter__(self):
        image_files = [os.path.join(self.dir, f) for f in os.listdir(self.dir) if
                       os.path.isfile(os.path.join(self.dir, f)) and f.endswith('.PNG')]
        print(image_files)
        for file in image_files:
            print(file)
            yield ImageIndividual(image=Image.open(file).convert('RGB'))
