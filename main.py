from genetic.basic_approach import BasicApproach

if __name__ == '__main__':

    genetic = BasicApproach(class_to_optimize="Vorfahrt")
    size = 20
    p = genetic.population(size)

    print(p)

    fitness_history = [genetic.grade(p), ]
    for i in range(10):
        print("------------------Grade: %f, Generation %s with %d individuals------------------" % (
            fitness_history[-1], str(i + 1), len(p)))
        p = genetic.evolve(p, retain=0.2)
        fitness_history.append(genetic.grade(p))
        print(p)

    print("------------------Finished Process Grade History------------------")

    print(fitness_history)

    print("------------------Final Population------------------")
    print(p)