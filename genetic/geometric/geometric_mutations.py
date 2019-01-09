from genetic.geometric.geometric_objects import GeometricObject
from genetic.geometric.circle_population_generator import generate_circle
class GeometricMutations:

    @staticmethod
    def mutate_circle(object: GeometricObject):
        return generate_circle(10, 5)