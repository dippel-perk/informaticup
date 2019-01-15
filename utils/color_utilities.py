import random as rd


class ColorUtilities:
    """
    Holds useful color related functions.
    """

    @staticmethod
    def interpolation_color_generator(color1, color2) -> tuple:
        """
        Generate random colors between color1 and color2.
        :param color1 The first color
        :param color2 The second color.
        :return Yields the random color.
        """
        # Get the difference along each axis
        d_red = color1[0] - color2[0]
        d_green = color1[1] - color2[1]
        d_blue = color1[2] - color2[2]

        while True:
            proportion = rd.uniform(0, 1)

            yield (color1[0] - int(d_red * proportion),
                   color1[1] - int(d_green * proportion),
                   color1[2] - int(d_blue * proportion))

    @staticmethod
    def random_color_generator() -> tuple:
        """
        Generates random colors.
        :return: Yields random colors.
        """
        while True:
            yield (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))
