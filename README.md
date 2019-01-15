# Genetic Algorithm for a Query Efficient Input-free Black-box Attack against a Neural Network with Limited Information

Previous work has shown that neural networks can be fooled by adversarial images. As part of the informaticup competition, we investigate a black box attack scenario where only limited query access (60 queries per minute) and partial output (top five classes) is given. This tool implements a genetic algorithm, which is capable of generating adversarial images which fool the target network.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

* Python version `>= 3.5`
* Install packages with `pip install -r requirements.txt`


Depending on your configuration you might have to replace the `python` command with `python3`. You can check if `python` executes the correct version with the following command

```
$ python --version
Python 3.5.x
```

To change the `python` command you can place an alias into `~/.bashrc` or `~/.bash_aliases` files

```
alias python=python3
```

## Usage

The program can be executed with the following command `python main.py`. There are several arguments which can be passed to change the behavior of the algorithm.

```
main.py [-h] -t TARGET [-c CONFIDENCE] [-s STEPS]
               [--population-size POPULATION_SIZE] [--retain-rate RETAIN_RATE]
               [--mutation-rate MUTATION_RATE]
               [--mutation-intensity MUTATION_INTENSITY]
               [--random-select-rate RANDOM_SELECT_RATE]
               [--gtsrb-image-path GTSRB_IMAGE_PATH] [-o OUT]
               (--rand | --color | --sample | --brute-force | --circle | --polygon | --gilogo | --tiles)
               {substitute} ...
```

When adding the substitute command at the **end** of the arguments you will have two additional parameters

```
main.py [...] substitute [-gpn GENETIC_POPULATION_SIZE]
                          [-gps GENETIC_POPULATION_STEPS]
```

### Arguments

1. `-t --target`
The target class name or target class id.
1. `-s --steps` The maximum number of steps, i.e. the number of the generations the algorithm will evolve.
1. `-c --confidence` The desired confidence. The algorithm will terminate, when the confidence is reached by at least one generated image. If no confidence is provided by the user, the algorithm will run the maximum number of steps. The confidence must be in range [0,1].
1. `--population-size` The desired population size. Every generation of the genetic algorithm will have this many individuals.
1. `--mutation-rate` The mutation rate which should be used.
1. `--mutation-intensity` The mutation intensity which should be used.
1. `--random-select-rate` The percentage of individuals which should be randomly added to the next generation.
1. `--gtsrb-image-path` The path to the German Traffic Sign Recognition Benchmark. The images of the benchmark are used in some population generators.
1. `--out` A complete history of all computed generations will be saved to an output directory. If no directory is provided, a standard directory is chosen.
1. What follows are the available population generators. The population generator might have influence on the mutation and crossover function. At least one population generator has to be chosen from the set of possibilities.
  1. `--rand` Generates a population of random image individuals.
  1. `--color` Generates a population with the same color distribution as some random training images.
  1. `--sample` Generates a population, which contains rearrangements of training set images.
  1. `--brute-force` Generates a population of random image individuals, while ensuring that each individual's classification contains the target class. To achieve this, new images are generated until the target class is part of the classification.
  1. `--circle` Generates population of geometric individuals, which are filled with random circles.
  1. `--polygon` Generates a population of geometric individuals which contain random polygons. We restricted the polygons to be triangles.
  1. `--gilogo` To come
  1. `--tiles` Generates a population with geometric individuals. Every individual is completely filled with so called tiles.
  1. `--snowflakes`

If the substitute command was added, the selected population generator will be the initial population generator for the substitute network. The tool will evolve this population to a state where the substitute network classifies the individuals to be most likely in the target class. The `substitute` has to be added **after** the arguments above. If added, the user has two additional arguments.

1. `-gpn --genetic-population-size` The population size which should be used by the substitute network. The size has to be larger than `--population-size` *n*, because we want to use the most fit *n* individuals of the substitute network as an initial population of the genetic algorithm. If the variable is not set, we use the size 3*n*.
1. `-gps --genetic-population-steps` Determines the amount of steps which should be performed on the substitute network.




### Help

For further help review the help pages
`python main.py -h` and `python main.py substitute -h` or contact us.


## Authors

* **Jonas Dippel**
* **Michael Perk**

## InformatiCup

[Website](https://gi.de/informaticup/)

[Aufgabenstellung](https://gi.de/fileadmin/GI/Hauptseite/Aktuelles/Wettbewerbe/InformatiCup/InformatiCup2019-Irrbilder.pdf)


Api Key: EiCheequoobi0WuPhai3saiLud4ailep
