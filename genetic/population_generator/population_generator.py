from tqdm import tqdm


class PopulationGenerator:
    """
    Generates a population of a given size.
    """

    def __init__(self, size: int):
        self.size = size
        self._progress_bar = None

    @property
    def progress_bar_total(self) -> int:
        """
        Returns the number of steps needed to fill a possible progess bar.
        :return: The number of steps.
        """
        return self.size

    @property
    def progress_bar_description(self) -> str:
        """
        Returns the description of the progress bar.
        :return: The description
        """
        return "Individuals"

    def register_progress_bar(self, progress_bar: tqdm) -> None:
        """
        Registers a progress bar. Only one progress bar can be registered at a time.
        :param progress_bar: The progress bar.
        :return: None
        """
        self._progress_bar = progress_bar

    def _progress_bar_step(self) -> None:
        """
        If a progress bar is registered, this method makes a single step.
        :return:
        """
        if self._progress_bar is not None:
            self._progress_bar.update(1)

    def __iter__(self):
        """
        This method should be overwritten by potential subclasses.
        This function should be called when the population generator is iterated. It yields a set image individuals.
        :return: Yields image individuals
        """
        raise NotImplementedError
