import random
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math
from tqdm import tqdm

class TrainingDataPreProcessing:
    def __init__(self):
        self.this_file_path = os.path.dirname(os.path.abspath(__file__))
        self.input_path = os.path.join(self.this_file_path, 'HW3_training_data')
        self.output_file = os.path.join(self.this_file_path, 'train_data.csv')

    def _write_csv(self, data:list):
        with open(self.output_file, 'w', encoding='utf-8', newline='') as csvWriter:
            writer = csv.writer(csvWriter)
            for row in data:
                writer.writerow(row)
    
    def execute(self):
        file_list = [_ for _ in os.listdir(self.input_path) if _.endswith(r'.txt') and _[:6] == "train_"] # 取出符合條件的檔案
        result_list = list()
        for file in file_list:
            file_path = os.path.join(self.input_path, file)
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    if '30.0000000' not in line:
                        line = line.replace('\n','')
                        line = line.split(' ')
                        result_list.append(line)

        self._write_csv(result_list)
        print('[TrainingDataPreProcessing] done.')

class GeneticAlgorithmRealNum:
    def __init__(self, pop_size, input_dimension, node_cnt, cross_rate, mutation_rate, max_gen):
        self.this_file_path = os.path.dirname(os.path.abspath(__file__))
        self.pop_size = pop_size               # 總染色體數
        self.gene_size = node_cnt + input_dimension*node_cnt + node_cnt + 1             # 染色體長度
        self.input_dimension = input_dimension
        self.node_cnt = node_cnt
        self.gene_limit = (                                                            # 設定上下限
        (0, node_cnt-1, 0, 1),
        (node_cnt, node_cnt*(input_dimension+1)-1, 0, 30),
        (node_cnt*(input_dimension+1), node_cnt*(input_dimension+2)-1, 0, 10),
        (self.gene_size-1 ,self.gene_size-1, 0, 1)
        )
        self.cross_rate = cross_rate           # 交配率
        self.mutation_rate = mutation_rate     # 突變率
        self.max_gen = max_gen                 # 最大迭代次數
        self.pop = []                          # 染色體群體
        self.fitness_value = []                # 個體分數
        self.mating_pool = []                  # 交配池

    def initialization(self):
        """產生初始population，不同index有不同的上下限
        """
        for _ in range(self.pop_size):
            temp_pop = []
            for j in range(self.gene_size):
                for start_index, end_index, lower, upper in self.gene_limit:
                    if j >= start_index and j <= end_index:
                        temp_pop.append(random.uniform(lower, upper))
                        break
            self.pop.append(temp_pop)
    
    def calc_fitness_value(self, group:str):
        csv_file_path = os.path.join(self.this_file_path, 'train_data.csv')
        with open(csv_file_path, 'r', encoding='utf-8', newline='') as CsvReader:
            CsvReader = csv.reader(CsvReader)
            train_data_list = list(CsvReader)
        total_train_data_cnt = len(train_data_list)

        if group == 'pop':
            population = self.pop
        elif group == 'mating_pool':
            population = self.mating_pool

        
        self.fitness_value = []
        for pop in population:
            error = 0
            for data in train_data_list:
                x= list(map(float, data[:3]))
                theta = float(data[-1])
                error += abs(theta - RBFNetwork(self.node_cnt, self.input_dimension, pop, x))
            fitness_value = 1 / (error / total_train_data_cnt)
            self.fitness_value.append(fitness_value)

    def selection_by_RWS(self):
        """用輪盤法抽選至交配池內
        """
        self.mating_pool.clear()
        if sum(self.fitness_value) == 0:
            self.mating_pool = self.pop.copy()
            return
        else:
            probability = [individual_value / sum(self.fitness_value) for individual_value in self.fitness_value]

        for _ in range(self.pop_size):
            random_pop = random.choices(self.pop, weights=probability)[0]
            random_pop = list(random_pop)
            self.mating_pool.append(random_pop)

    def crossover(self):
        """交配池內的染色體進行交配
        """
        shuffle_mating_pool = np.random.permutation(self.mating_pool) # 打亂mating_pool順序

        for i in range(0, len(shuffle_mating_pool), 2):
            cr = random.uniform(0,1)
            if cr < self.cross_rate:
                alpha = random.uniform(0,1)
                spring_1 = alpha * shuffle_mating_pool[i] + (1-alpha)*shuffle_mating_pool[i+1]
                spring_2 = (1-alpha)*shuffle_mating_pool[i] + alpha * shuffle_mating_pool[i+1]
                shuffle_mating_pool[i] = spring_1
                shuffle_mating_pool[i+1] = spring_2
                
        shuffle_mating_pool = [list(x) for x in shuffle_mating_pool]  # numpy.ndarray轉為list
        self.mating_pool = shuffle_mating_pool

    def mutation(self):
        """交配池內的染色體進行突變
        """
        for pop in self.mating_pool:
            for index in range(self.gene_size):
                mr = random.uniform(0,1)
                if mr < self.mutation_rate:
                    for start, end, lower, upper in self.gene_limit:
                        if index >= start and index <= end:
                            pop[index] = random.uniform(lower,upper)
                            break

    def execute(self) -> list:
        self.initialization()
        self.calc_fitness_value('pop')
        gen_best_model = {}
        for gen in tqdm(range(1, self.max_gen+1),ncols=50):
            temp_dict = {}
            self.selection_by_RWS()
            self.crossover()
            self.mutation()
            self.calc_fitness_value('mating_pool')
            self.pop = self.mating_pool.copy()  # replace

            best_model_index = self.fitness_value.index(min(self.fitness_value))
            best_model = self.pop[best_model_index]
            best_fitness_value = min(self.fitness_value)

            temp_dict['model'] = best_model
            temp_dict['fitness'] = best_fitness_value
            gen_best_model[gen] = temp_dict

        # 取出最佳結果回傳
        min_fitness_model = min(gen_best_model.values(), key=lambda x: x['fitness'])['model']

        return min_fitness_model

class Car:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
        self.position_x = 0
        self.position_y = 0
        self.orientation = 90 # 90 ~ -270

    def calc_next_step(self, theta):
        next_x = self.position_x + math.cos(math.radians(self.orientation + theta)) + math.sin(math.radians(theta))*math.sin(math.radians(self.orientation))
        next_y = self.position_y + math.sin(math.radians(self.orientation + theta)) - math.sin(math.radians(theta))*math.cos(math.radians(self.orientation))
        self.position_x, self.position_y = next_x, next_y

        self.orientation -= self.orientation - math.degrees(math.asin(2*math.sin(math.radians(theta))/self.length))
        if self.orientation < -270:
            self.orientation += 360
        if self.orientation > 90:
            self.orientation -= 360


def RBFNetwork(node_cnt:int, input_dimension:int, chromosome:list[float], x:list[float]) -> float:
    Theta = 0
    for i in range(node_cnt):
        m_start_index = node_cnt+i*input_dimension
        m_end_index = m_start_index + input_dimension
        sigma_start_index = node_cnt + input_dimension*node_cnt

        W_i = chromosome[i]
        m_vector = np.array(chromosome[m_start_index:m_end_index])

        sigma = chromosome[sigma_start_index+i]
        phi_i_of_x = np.exp(-(np.linalg.norm(np.array(x) - m_vector)) / (2*sigma**2))
        Theta += W_i * phi_i_of_x
    
    Theta += chromosome[-1]

    # result = (40 - (-40)) * (float(Theta) - 0) / (1 - 0) + (-40) # 0~1 -> -40~40
    # return result
    return float(Theta)

def save_model(data:list[float], path:str='best_model.csv'):
    with open(path,'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def load_model(path:str='best_model.csv') -> list[float]:
    try:
        with open(path, 'r', newline='') as file:
            reader = csv.reader(file)
            model = next(reader)
            model = [float(item) for item in model]
            return model
    except FileNotFoundError:
        print(f"[Errno 2] No such file or directory: '{path}'")


def calc_distance_from_2_point(point1:tuple, point2:tuple) -> float:
    # 給兩點，求距離
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    distance = math.hypot(x2-x1,y2-y1)

    return distance

def calc_line_with_point_and_angle(point:tuple, angle:float) -> tuple[float]:
    # 給點和角度，求線
    x,y = point
    theta = math.radians(angle)
    
    if angle % 180 == 90: # 垂直
        A = 1
        B = 0
        C = -x
    else:
        m = math.tan(theta)
        b = y - m * x
        A = -m
        B = 1
        C = -b

    return A,B,C

def calc_line_with_2point(point1:tuple, point2:tuple) -> tuple[float]:
    # 給兩點，求線
    x1,y1 = point1
    x2,y2 = point2
    if x1 == x2: # 垂直
        A = 1
        B = 0
        C = -x1
    else:
        m = (y2-y1) / (x2-x1)
        b = y1 - m * x1
        A = -m
        B = 1
        C = -b

    return A,B,C

def find_intersection(line1:tuple[float], line2:tuple[float]) -> None|tuple[float]:
    # 給兩線，求交點
    A1,B1,C1 = line1
    A2,B2,C2 = line2

    A = np.array([[A1, B1], [A2, B2]])
    B = np.array([-C1, -C2])

    try:
        solution = np.linalg.solve(A, B)
        x, y = solution

        return (x, y)
    except np.linalg.LinAlgError:  # 無解或無唯一解 (平行,重合)
        return None
    
def direction_unit_vector(angle:float) -> tuple[float]:
    # 給角度，求單位向量
    theta = math.radians(angle)
    unit_vector_x = math.cos(theta)
    unit_vector_y = math.sin(theta)

    return unit_vector_x, unit_vector_y

def calc_dF_dL_dR(walls:list[tuple], car:Car):
    # 計算dF、dL、dR的距離
    front = car.orientation
    left = car.orientation + 90
    right = car.orientation - 90

    if left > 90:   # 調整範圍至+90 ~ -270
        left -= 360
    if right < -270:
        right += 360

    directions = [front, left, right]
    result_distance = []

    for d in directions:
        # 取得車子和牆壁的交點
        intersection = []
        for wall in walls:
            car_line = calc_line_with_point_and_angle((car.position_x,car.position_y), d)
            wall_line = calc_line_with_2point(wall[0], wall[1])
            
            intersection.append(find_intersection(car_line, wall_line))
        print('intersection: ',intersection)
        # 濾除範圍外的交點
        cnt = 0
        filtered_intersection = []
        for point in intersection:
            if point is not None:
                point = (round(point[0],2), round(point[1],2))  # TEST
                if point[0] >= walls[cnt][0][0] and point[0] <= walls[cnt][1][0] and \
                point[1] >= walls[cnt][0][1] and point[1] <= walls[cnt][1][1]: # 確認交點是否在線段範圍內
                    
                    filtered_intersection.append(point)
            cnt += 1
        
        print('filtered_intersection: ', filtered_intersection)

        # 保留向量內積大於0的交點
        car_direction_vector = direction_unit_vector(d)
        result_point = []
        for point in filtered_intersection:
            vector_x = point[0] - car.position_x
            vector_y = point[1] - car.position_y
            vector = (vector_x, vector_y)

            dot_product = np.dot(car_direction_vector, vector)

            if dot_product > 0:
                result_point.append(point)

        # 取出距離最近的
        temp_distance = []
        for point in result_point:
            result = calc_distance_from_2_point(point, (car.position_x,car.position_y))
            temp_distance.append(result)
        if temp_distance:
            result_distance.append(min(temp_distance))
        else:
            result_distance.append(float('inf'))

    print('result_distance', result_distance)
    return result_distance


if __name__ =='__main__':

    INPUT_DIMENSION = 3
    NODE_CNT = 7

    CAR_RADIUS = 3
    CAR_LENGTH = 1

    POP_SIZE = 100
    CROSS_RATE = 0.8
    MUTATION_RATE = 0.2
    MAX_GEN = 100

    walls = [((-6, 0), (-6, 22)),
                ((-6, 22), (18, 22)),
                ((18, 22), (18, 37)),
                ((18, 50), (30, 50)),  # 終點邊界  原始為37 -> 50
                ((30, 10), (30, 37)),
                ((6, 10), (30, 10)),
                ((6, 0), (6, 10)),
                ((-6, -10), (6, -10))]  # 起點邊界  原始為0 -> -10
    
    goal_position = (24, 37)

    # 若已有模型，此段可不跑
    # ===========training data 建立模型===========start
    TrainingDataPreProcessing().execute()

    best_model = GeneticAlgorithmRealNum(pop_size=POP_SIZE,
                                 input_dimension=INPUT_DIMENSION,
                                 node_cnt=NODE_CNT,
                                 cross_rate=CROSS_RATE,
                                 mutation_rate=MUTATION_RATE,
                                 max_gen=MAX_GEN).execute()

    save_model(best_model, 'best_model.csv')
    # ===========training data 建立模型===========end

    best_model = load_model('best_model.csv')

    car = Car(radius=CAR_RADIUS,
              length=CAR_LENGTH)
    
    fig, ax = plt.subplots()
    plt.ion()
    for wall in walls: # 畫出地圖邊界
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black')
    
    plt.xlim(-10, 40)
    plt.ylim(-5, 40)
    plt.xlabel('X Location')
    plt.ylabel('Y Location')
    plt.grid(True)  # 顯示格線

    it = 0 # iteration
    while (int(car.position_x), int(car.position_y)) != goal_position or it < 100 or min(result_distance) <= car.radius:
        plt.plot(car.position_x, car.position_y, marker='s', color='r', fillstyle='none')
        plt.pause(3) # 畫面更新間隔
        result_distance = calc_dF_dL_dR(walls, car)

        Theta = RBFNetwork(node_cnt=NODE_CNT,
                input_dimension=INPUT_DIMENSION,
                chromosome=best_model, 
                x = result_distance)

        car.calc_next_step(theta=Theta)
        # TODO: 碰撞半徑問題 car.radius
        # TODO: 角度是以車子為基準還是xy座標為基準
        # TODO: theta rescaling
        # TODO: 方向盤角度y軸為0，逆時針為+，順時針為-
        it += 1

    plt.show()