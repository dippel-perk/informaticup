from genetic.population_generator.random_population_generator import RandomPopulationGenerator
from road_sign_class_mapper import RoadSignClassMapper
from PIL import Image
import os
from collections import defaultdict
import random as rd
import numpy as np
from utils.image_utilities import ImageUtilities


class TrainColorPopulationGenerator(RandomPopulationGenerator):
        def __init__(self, size: int, target_class: int, image_dir: str):
            super().__init__(size=size)
            color_distribution = defaultdict(int)

            directory = os.fsencode(os.path.join(image_dir, str(target_class).zfill(5)))
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if not filename.endswith('.png'):
                    continue
                img = Image.open(os.path.join(directory, file)).convert('RGB')
                colors = img.getcolors(img.size[0]*img.size[1])
                for count, color in colors:
                    color_distribution[color] += count

            self._colors, self._probabilities = zip(*color_distribution.items())

            total = sum(self._probabilities)
            self._probabilities = [x / total for x in self._probabilities]

        def _generate_noise(self):
            img, pixel_count = ImageUtilities.get_empty_image()

            pixel_indices =  list(np.random.choice(list(range(len(self._colors))), p=self._probabilities, size=pixel_count))
            pixel_data = [self._colors[idx] for idx in pixel_indices]
            img.putdata(pixel_data)

            return img

        def _get_pixel_value(self):
            decision_val = rd.random()
            if decision_val < 0.2:
                return rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255)
            else:
                color_index = np.random.choice(list(range(len(self._colors))), p=self._probabilities)
                return self._colors[color_index]



