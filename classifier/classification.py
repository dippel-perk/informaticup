import string
from typing import List


class Class:
    """
    Stores the confidence result for a certain class.
    """

    def __init__(self, name: string, confidence: string):
        self.name = name
        self.confidence = float(confidence)

    def __repr__(self):
        return "%s %f" % (self.name, self.confidence)


class ImageClassification():
    """
    Stores the classification result for a certain image.
    """

    def __init__(self, classes: List[Class]):
        self.classes = classes

    def share_classes(self, other) -> bool:
        """
        Determines if the classification shares at least one class with the given image classification
        :param other: The other image classification
        :return: True if the classification results share a class, False otherwise
        """
        own_names = (cl.name for cl in self.classes)
        other_names = (cl.name for cl in other.classes)
        return any(name in own_names for name in other_names)

    def value_for_class(self, class_name) -> float:
        return next((x.confidence for x in self.classes if str(x.name) == str(class_name)), 0)

    def __repr__(self):
        return "Image Classification %s\n" % ('\n'.join(cl.__repr__() for cl in self.classes))
