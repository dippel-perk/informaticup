import string
class ImageClassification:
    def __init__(self, file, classes):
        self.file = file
        self.classes = classes

    def share_classes(self, other):
        own_names = (cl.name for cl in self.classes)
        other_names = (cl.name for cl in other.classes)
        return any(name in own_names for name in other_names)

    def value_for_class(self, class_name):
        return next((x.confidence for x in self.classes if x.name == class_name), 0)

    def __repr__(self):
        return "Image Classification %s\n%s\n" % (self.file, '\n'.join(cl.__repr__() for cl in self.classes))

class Class:
    def __init__(self, name : string, confidence : string):
        self.name = name
        self.confidence = float(confidence)

    def __repr__(self):
        return "%s %f" % (self.name, self.confidence)