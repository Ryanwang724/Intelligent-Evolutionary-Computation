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

POP_SIZE = 100   # even number for crossover
GENE_SIZE = 10
CROSS_RATE = 0.6 # 0.8 to 0.6
MUTATION_RATE = 0.2
MAX_GEN = 100
UPPER_BOUNDARY = 3
LOWER_BOUNDARY = -3
EXTREMUM = 'min'

if __name__ == '__main__':
    # ga_rosenbrock = GeneticAlgorithm(pop_size = POP_SIZE, 
    #                     gene_size = GENE_SIZE*2, 
    #                     cross_rate = CROSS_RATE, 
    #                     mutation_rate = MUTATION_RATE,
    #                     max_gen = MAX_GEN,
    #                     func = rosenbrock,
    #                     upper = UPPER_BOUNDARY,
    #                     lower = LOWER_BOUNDARY,
    #                     extremum = 'min',
    #                     show_mode = True)
    # ga_rosenbrock.execute()

    POP_SIZE = 100   # even number for crossover
    GENE_SIZE = 14
    CROSS_RATE = 0.8
    MUTATION_RATE = 0.4
    MAX_GEN = 200
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
                        show_mode = False)
    ga_eggholder.execute()
