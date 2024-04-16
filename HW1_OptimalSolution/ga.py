import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from GeneticAlgorithm import GeneticAlgorithm

def rosenbrock(x1, x2):
    return 100*(x2-x1**2)**2 + (1-x1)**2

def eggholder(x, y):
    return -(y+47)*np.sin(np.sqrt(abs(x/2+(y+47)))) - x*np.sin(np.sqrt(abs(x-(y+47))))

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def rastrigin(x1, x2):
    return 10*2 + (x1**2 - 10*np.cos(2*np.pi*x1)) + (x2**2 - 10*np.cos(2*np.pi*x2))

if __name__ == '__main__':
    run_select = 1      # 1:rosenbrock 2:eggholder 3:himmelblau 4:rastrigin
    if run_select == 1:
        POP_SIZE = 200   # even number for crossover
        GENE_SIZE = 10
        CROSS_RATE = 0.6 # 0.8 to 0.6
        MUTATION_RATE = 0.4
        MAX_GEN = 500
        UPPER_BOUNDARY = 3
        LOWER_BOUNDARY = -3
        EXTREMUM = 'min'

        ga_rosenbrock = GeneticAlgorithm(pop_size = POP_SIZE, 
                            gene_size = GENE_SIZE*2, 
                            cross_rate = CROSS_RATE, 
                            mutation_rate = MUTATION_RATE,
                            max_gen = MAX_GEN,
                            func = rosenbrock,
                            upper = UPPER_BOUNDARY,
                            lower = LOWER_BOUNDARY,
                            extremum = 'min',
                            show_mode = True)
        ga_rosenbrock.execute()
    elif run_select == 2:
        POP_SIZE = 200   # even number for crossover
        GENE_SIZE = 16
        CROSS_RATE = 0.8
        MUTATION_RATE = 0.2
        MAX_GEN = 300
        UPPER_BOUNDARY = 1000
        LOWER_BOUNDARY = -1000
        EXTREMUM = 'min'

        ga_eggholder = GeneticAlgorithm(pop_size = POP_SIZE, 
                            gene_size = GENE_SIZE*2, 
                            cross_rate = CROSS_RATE, 
                            mutation_rate = MUTATION_RATE,
                            max_gen = MAX_GEN,
                            func = eggholder,
                            upper = UPPER_BOUNDARY,
                            lower = LOWER_BOUNDARY,
                            extremum = 'min',
                            show_mode = True)
        ga_eggholder.execute()
    elif run_select == 3:
        POP_SIZE = 200   # even number for crossover
        GENE_SIZE = 16
        CROSS_RATE = 0.8
        MUTATION_RATE = 0.2
        MAX_GEN = 300
        UPPER_BOUNDARY = 5
        LOWER_BOUNDARY = -5
        EXTREMUM = 'min'

        ga_himmelblau = GeneticAlgorithm(pop_size = POP_SIZE, 
                            gene_size = GENE_SIZE*2, 
                            cross_rate = CROSS_RATE, 
                            mutation_rate = MUTATION_RATE,
                            max_gen = MAX_GEN,
                            func = himmelblau,
                            upper = UPPER_BOUNDARY,
                            lower = LOWER_BOUNDARY,
                            extremum = 'min',
                            show_mode = True)
        ga_himmelblau.execute()
    elif run_select == 4:
        POP_SIZE = 200   # even number for crossover
        GENE_SIZE = 16
        CROSS_RATE = 0.8
        MUTATION_RATE = 0.2
        MAX_GEN = 300
        UPPER_BOUNDARY = 5
        LOWER_BOUNDARY = -5
        EXTREMUM = 'min'

        ga_rastrigin = GeneticAlgorithm(pop_size = POP_SIZE, 
                            gene_size = GENE_SIZE*2, 
                            cross_rate = CROSS_RATE, 
                            mutation_rate = MUTATION_RATE,
                            max_gen = MAX_GEN,
                            func = rastrigin,
                            upper = UPPER_BOUNDARY,
                            lower = LOWER_BOUNDARY,
                            extremum = 'min',
                            show_mode = True)
        ga_rastrigin.execute()