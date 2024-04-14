import random
import numpy as np
import math

class GeneticAlgorithm:
    def __init__(self, pop_size, gene_size, cross_rate, mutation_rate, max_gen, func, upper, lower):
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.max_gen = max_gen
        self.pop = []
        self.fitness_value = []
        self.mating_pool = []

        self.func = func
        self.upper = upper
        self.lower = lower

    def decode(self, pop):
        decimal_1 = ''
        decimal_2 = ''

        for index in range(self.gene_size // 2):  # 前半段
            decimal_1 += str(pop[index])
        for index in range(self.gene_size // 2):  # 後半段
            decimal_2 += str(pop[index + self.gene_size // 2])

        decimal_1 = int(str(decimal_1),2)                          # binary to decimal
        decimal_2 = int(str(decimal_2),2)

        decimal_1 = float(decimal_1) / (2 ** (self.gene_size//2))  # normalization to 0 ~ 1
        decimal_2 = float(decimal_2) / (2 ** (self.gene_size//2))  # normalization to 0 ~ 1

        result_1 = decimal_1 * (self.upper-self.lower) + self.lower
        result_2 = decimal_2 * (self.upper-self.lower) + self.lower

        return [result_1, result_2]

    def fitness(self):
        pass

    def initialization(self):
        for i in range(self.pop_size):
            temp_pop = []
            for j in range(self.gene_size):
                temp_pop.append(random.choice([0, 1]))
            self.pop.append(temp_pop)

    def evaluation(self):
        for pop in self.pop:
            result = self.decode(pop)
            value = self.func(result[0], result[1])
            self.fitness_value.append(value)

    def selection(self):
        total_fitness_value = sum(1/x for x in self.fitness_value)
        probability = [(1 / x) / total_fitness_value for x in self.fitness_value] # 最小化問題=>數字越小機率越大
        for _ in range(self.pop_size):
            random_pop = random.choices(self.pop, weights=probability)[0]
            self.mating_pool.append(random_pop)

    def crossover(self):
        def swap_func(a,b):
            temp = a
            a = b
            b = temp
            return a,b
        
        shuffle_mating_pool = np.random.permutation(self.mating_pool)
        for i in range(0, len(shuffle_mating_pool), 2):
            cr = random.uniform(0,1)
            if cr < self.cross_rate:
                cross_location = np.random.uniform(1, self.gene_size)
                shuffle_mating_pool[i][:cross_location], shuffle_mating_pool[i+1][:cross_location] = swap_func(shuffle_mating_pool[i][:cross_location], shuffle_mating_pool[i+1][:cross_location])
        

    def mutation(self):
        pass

    def execute(self):
        pass