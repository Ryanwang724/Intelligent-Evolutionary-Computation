import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import os

class GeneticAlgorithm:
    def __init__(self, pop_size, gene_size, cross_rate, mutation_rate, max_gen, func, upper, lower, extremum, show_mode):
        self.pop_size = pop_size               # 總染色體數
        self.gene_size = gene_size             # 染色體
        self.cross_rate = cross_rate           # 交配率
        self.mutation_rate = mutation_rate     # 突變率
        self.max_gen = max_gen                 # 最大迭代次數
        self.pop = []                          # 染色體群體
        self.fitness_value = []                # 個體分數
        self.mating_pool = []                  # 交配池

        self.func = func                       # 欲執行的函式名稱
        self.upper = upper                     # 搜索範圍上限
        self.lower = lower                     # 搜索範圍下限
        self.extremum = extremum               # 找最大或最小值
        self.best_result = []                  # 每一代的最佳fitness_value
        self.best_parameter = []               # 每一代最佳fitness_value的參數

        self.show_mode = show_mode             # 執行時是否在終端顯示執行log

    def decode(self, pop:list):
        """將染色體內的二進制資料轉為搜索範圍內的值

        Args:
            pop (list): 一次一個pop進來，格式為[1,0,0,1,1]

        Returns:
            list: [x1,y1]
        """
        decimal_1 = ''
        decimal_2 = ''
        # 因有兩個參數，故將基因分成前後兩段
        for index in range(self.gene_size // 2):  # 前半段
            decimal_1 += str(pop[index])
        for index in range(self.gene_size // 2):  # 後半段
            decimal_2 += str(pop[index + self.gene_size // 2])

        decimal_1 = int(str(decimal_1),2)                           # binary to decimal
        decimal_2 = int(str(decimal_2),2)

        decimal_1 = float(decimal_1) / (2 ** (self.gene_size//2))   # normalization to 0 ~ 1
        decimal_2 = float(decimal_2) / (2 ** (self.gene_size//2))

        result_1 = decimal_1 * (self.upper-self.lower) + self.lower # 線性轉換成上下限範圍內的值
        result_2 = decimal_2 * (self.upper-self.lower) + self.lower

        return [result_1, result_2]

    def initialization(self):
        """產生初始population
        """
        for i in range(self.pop_size):
            temp_pop = []
            for j in range(self.gene_size):
                temp_pop.append(random.choice([0, 1]))
            self.pop.append(temp_pop)

    def evaluation(self, group:str):
        """計算每個pop帶入function內的值

        Args:
            group (str): 輸入要計算哪個群體:'pop' or 'mating_pool'
        """
        if group == 'pop':
            population = self.pop
        elif group == 'mating_pool':
            population = self.mating_pool

        self.fitness_value = []
        for pop in population:
            result = self.decode(pop)
            value = self.func(result[0], result[1])
            self.fitness_value.append(value)

    def selection_by_RWS(self):
        """利用softmax計算出個別的機率後，用輪盤法抽選至交配池內
        """
        def softmax_max(x):
            x = [xi-max(x) for xi in x]
            e_x = [np.exp(-xi) for xi in x]
            return [x/sum(e_x) for x in e_x]
        def softmax_min(x):
            x = [xi-max(x) for xi in x]
            e_x = [np.exp(xi) for xi in x]
            return [x/sum(e_x) for x in e_x]

        if self.extremum == 'min' and sum(self.fitness_value) == 0:
            self.mating_pool = self.pop
        elif self.extremum == 'min':
            probability = softmax_min(self.fitness_value)
        elif self.extremum == 'max':
            probability = softmax_max(self.fitness_value)

        for _ in range(self.pop_size):
            random_pop = random.choices(self.pop, weights=probability)[0]
            random_pop = list(random_pop)
            self.mating_pool.append(random_pop)

    def crossover(self):
        """交配池內的染色體進行交配
        """
        def swap_func(a,b):
            temp = a
            a = b
            b = temp
            return a,b
        
        shuffle_mating_pool = np.random.permutation(self.mating_pool) # 打亂mating_pool順序
        shuffle_mating_pool = [list(x) for x in shuffle_mating_pool]  # 轉為list

        for i in range(0, len(shuffle_mating_pool), 2):
            cr = random.uniform(0,1)
            if cr < self.cross_rate:
                x_or_y = random.choice([0,1]) # 決定要交換x或y  0:x, 1:y
                cross_location = int(np.random.uniform(1, self.gene_size//2 + 1)) # 決定要從哪邊開始交換(uniform)
                if x_or_y == 0:
                    shuffle_mating_pool[i][cross_location-1:self.gene_size//2], shuffle_mating_pool[i+1][cross_location-1:self.gene_size//2] = swap_func(shuffle_mating_pool[i][cross_location-1:self.gene_size//2], shuffle_mating_pool[i+1][cross_location-1:self.gene_size//2])
                else:
                    shuffle_mating_pool[i][cross_location-1+self.gene_size//2:], shuffle_mating_pool[i+1][cross_location-1+self.gene_size//2:] = swap_func(shuffle_mating_pool[i][cross_location-1+self.gene_size//2:], shuffle_mating_pool[i+1][cross_location-1+self.gene_size//2:])
                
        self.mating_pool = shuffle_mating_pool

    def mutation(self):
        """交配池內的染色體進行突變
        """
        for p in self.mating_pool:
            for i in range(self.gene_size):
                mr = random.uniform(0,1)
                if mr < self.mutation_rate:
                    if p[i] == 0:
                        p[i] = 1
                    else:
                        p[i] = 0

    def clean(self):
        """清空交配池
        """
        self.mating_pool = []

    def plot_iteration_result(self):
        """畫出每次迭代的最佳結果
        """
        X = [x for x in range(1, self.max_gen+1)]
        Y = self.best_result
        plt.subplot(1, 2, 1)
        plt.plot(X, Y, label='best_fitness_value')
        plt.xlabel(f'Generation: 1~{self.max_gen}')
        plt.ylabel('fitness_value')
        plt.legend()

    def plot_best_result(self):
        """畫出方程式圖形與找到的最佳解位置(紅色星號)
        """
        x1 = np.linspace(self.lower, self.upper, 100)
        x2 = np.linspace(self.lower, self.upper, 100)
        X1, X2 = np.meshgrid(x1,x2)
        all_value = self.func(X1,X2)
        
        if self.func.__name__ == 'rosenbrock':  # 不同function有不同的畫圖方式
            plt.subplot(1, 2, 2)
            plt.contourf(X1, X2, all_value, norm=LogNorm(), levels=50, alpha=0.9)
        elif self.func.__name__ == 'eggholder':
            plt.subplot(1, 2, 2)
            plt.contourf(X1, X2, all_value, levels=500)
        elif self.func.__name__ == 'himmelblau':
            plt.subplot(1, 2, 2,projection='3d')
            plt.contourf(X1, X2, all_value, levels=500,cmap=cm.jet)
        elif self.func.__name__ == 'rastrigin':
            plt.subplot(1, 2, 2,projection='3d')
            plt.contourf(X1, X2, all_value, levels=500,cmap=cm.jet)

        if self.extremum == 'min':
            best_value = min(self.best_result)
        elif self.extremum == 'max':
            best_value = max(self.best_result)
        index = self.best_result.index(best_value)
        para = self.best_parameter[index]
        plt.plot(para[0], para[1], 'r*')
        print(f'best para: {para[0]:.4f}, {para[1]:.4f}')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar()
        plt.title(f'GA {self.func.__name__} function')

        if not os.path.isdir(f'./img/{self.func.__name__}'):  # 存檔
            os.makedirs(f'./img/{self.func.__name__}')
        plt.savefig(f'./img/{self.func.__name__}/p{self.pop_size}g{int(self.gene_size/2)}c{self.cross_rate}m{self.mutation_rate}max{self.max_gen}.jpg')
        plt.show()
    def execute(self):
        self.initialization()
        self.evaluation('pop')
        for gen in range(1, self.max_gen+1):
            self.selection_by_RWS()
            self.crossover()
            self.mutation()
            self.evaluation('mating_pool')

            if self.extremum == 'max':
                best_fitness = max(self.fitness_value)
            elif self.extremum == 'min':
                best_fitness = min(self.fitness_value)
            self.best_result.append(best_fitness)

            index = self.fitness_value.index(best_fitness)   # 取得index
            parameter = self.decode(self.mating_pool[index]) # 轉為real value
            self.best_parameter.append(parameter)
            if self.show_mode:
                print(f'Generation {gen}:')
                print(f"    best_value: {best_fitness:.4f}    parameter: {parameter[0]:.4f}, {parameter[1]:.4f}")
            self.pop = self.mating_pool  # replace
            self.clean()
        
        self.plot_iteration_result()
        self.plot_best_result()