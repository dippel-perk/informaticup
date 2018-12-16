from image_utilities import ImageUtilities
from api import NeuralNetworkAPI
from classification import ImageClassification
from random import randint, random, sample
from PIL import Image


class GeneticAlgorithm:

    def __init__(self, class_to_optimize):
        self.api = NeuralNetworkAPI()
        self.class_to_optimize = class_to_optimize

    def individual(self):
        pass

    def population(self, count):
       return [self.individual() for i in range(count)]

    def fitness(self, individual):
        pass

    def grade(self, population):
        summed = sum(self.fitness(x) for x in population)
        return summed / len(population)

    def evolve(self, population, retain=0.2, random_select=0.05, mutate=0.01):
        pass

class Individual:
    def __init__(self, api: NeuralNetworkAPI, image : Image, classification : ImageClassification = None):
        self.api = api
        self.image = image
        self.classification = classification

    def classify(self, force_recomputation = False):
        if not self.classification or force_recomputation:
            file = ImageUtilities.save_image_to_tempfile(self.image)
            self.classification = self.api.classify(file)

    def mutate(self, position_to_mutate):
        ImageUtilities.mutate_pixel(self.image, position_to_mutate)

    def combine(self, individual):
        image = ImageUtilities.combine_images(self.image, individual.image)
        return Individual(api=self.api, image=image)

    def combinable(self, individual):
        self.classify()
        individual.classify()

        return self.classification.share_classes(individual.classification)

    def __len__(self):
        width, height = self.image.size
        return width*height

    def __repr__(self):
        return self.classification.__repr__()


class FirstApproach(GeneticAlgorithm):

    def individual(self):
        individual = Individual(api=self.api, image=ImageUtilities.generate_random_noise())
        individual.classify()

        while individual.classification.value_for_class(self.class_to_optimize) == 0:
            individual = Individual(api=self.api, image=ImageUtilities.generate_random_noise())
            individual.classify()

        return individual


    def fitness(self, individual):
        individual.classify()
        return individual.classification.value_for_class(self.class_to_optimize)

    def grade(self, population):
        return max(self.fitness(x) for x in population)

    def evolve(self, population, retain=0.2, random_select=0.05, mutate=0.01):
        graded = sorted(population, key = lambda x: self.fitness(x))

        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]

        # randomly add other individuals to
        # promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if mutate > random():
                pos_to_mutate = randint(0, len(individual)-1)
                individual.mutate(pos_to_mutate)
                individual.classify(force_recomputation=True) #recompute classification
                
        # crossover parents to create children
        desired_length = len(population) - len(parents)
        children = []

        while len(children) < desired_length:
            male, female = sample(parents, 2)

            if male.combinable(female):
                children.append(male.combine(female))

        parents.extend(children)

        return parents

if __name__ == '__main__':

    genetic = FirstApproach("Zulässige Höchstgeschwindigkeit (30)")
    size = 10
    p = genetic.population(size)

    print(p)

    fitness_history = [genetic.grade(p),]
    for i in range(6):
        print("------------------Grade: %f, Generation %s with %d individuals------------------" % (fitness_history[-1], str(i+1),len(p)))
        p = genetic.evolve(p, retain=0.2)
        fitness_history.append(genetic.grade(p))
        print(p)


    print("------------------Finished Process Grade History------------------")

    print(fitness_history)

    print("------------------Final Population------------------")
    print(p)