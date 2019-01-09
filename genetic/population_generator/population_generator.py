class PopulationGenerator:
    """
    Generates a population of a given size.
    """

    def __init__(self, size : int):
        self.size = size

    def __iter__(self):
        raise NotImplementedError
