import random
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
from copy import deepcopy

INPUT_DIMENSION = 3
NODE_CNT = 7

CAR_RADIUS = 3
CAR_LENGTH = 2    # 5.8

POP_SIZE = 100
CROSS_RATE = 0.6
MUTATION_RATE = 0.2
MAX_GEN = 100

TRAIN = True
TEST = True

# lambda function
dist = lambda x: np.sum(x ** 2, axis = -1)                            # Mahalanobis distance
theta_scale = lambda theta: (theta + 40) / 80.0                       # [-40, 40] -> [0, 1] for read training data
output_scale = lambda output: ((output * 80.0) - 40)                  # [0, 1] -> [-40, 40] 


def RBF_network_multiple_params(params:np.ndarray, inputs:np.ndarray) -> np.ndarray:
    weights = params[:, :NODE_CNT]  # shape: (POP_SIZE, 7)
    means = params[:, NODE_CNT: NODE_CNT + NODE_CNT * INPUT_DIMENSION].reshape(-1, NODE_CNT, INPUT_DIMENSION)  # shape: (POP_SIZE, 7, 3)
    stds = params[:, -1 - NODE_CNT: -1]  # shape: (POP_SIZE, 7)
    bias = params[:, -1]  # scaler      shape: (POP_SIZE,)

    inputs = inputs[:, None, None]  # shape: (Batch, 1, 1, 3)

    return (np.sum(np.exp(-dist(inputs - means) / (2 * (stds ** 2))) * weights, axis = -1) + bias)

def RBF_network_single_params(params:np.ndarray, inputs:np.ndarray) -> np.ndarray:
    weights = params[:NODE_CNT][None].T  # shape: (7, 1)
    means = params[NODE_CNT: NODE_CNT + NODE_CNT * INPUT_DIMENSION].reshape(NODE_CNT, INPUT_DIMENSION)  # shape: (7, 3)
    stds = params[-1 - NODE_CNT: -1]  # shape: (7,)
    bias = params[-1]  # scaler  (POP_SIZE,)

    inputs = inputs[:, None]  # shape: (Batch, 1, 1, 3)

    return (np.exp(-dist(inputs - means) / (2 * (stds ** 2))) @ weights + bias).T[0]

class GeneticAlgorithmRealNum:
    def __init__(self, pop_size:int, input_dimension:int, node_cnt:int, cross_rate:float, mutation_rate:float, max_gen:int):
        self.this_file_path = os.path.dirname(os.path.abspath(__file__))
        self.pop_size = pop_size               # 總染色體數
        self.gene_size = node_cnt + input_dimension*node_cnt + node_cnt + 1            # 染色體長度
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

    def load_data(self) -> tuple[np.ndarray]:
        training_data_path = 'HW3_training_data'
        file_list = [_ for _ in os.listdir(training_data_path) if _.endswith(r'.txt') and _[:6] == "train_"] # 取出符合條件的檔案

        X, Y =[], []
        for file in file_list:
            with open(f'{training_data_path}/{file}', 'r') as f:
                content = [list(map(float, line[:-1].split(' '))) for line in f if '30.0000001' not in line]

            X += [list(map(float, data[:3])) for data in content]
            Y += [theta_scale(float(data[-1])) for data in content]

        return np.array(X), np.array(Y)

    def initialization(self) -> np.ndarray:
        population = []
        for _ in range(self.pop_size):
            temp_pop = []
            for j in range(self.gene_size):
                for start_index, end_index, lower, upper in self.gene_limit:
                    if j >= start_index and j <= end_index:
                        temp_pop.append(random.uniform(lower, upper))
                        break
            population.append(temp_pop)
        
        return np.array(population)   # shape: (POP_SIZE, self.gene_size)
    
    def calc_fitness_value(self, group:np.ndarray) -> np.ndarray:
        data_X, data_Y = self.load_data()

        population = np.array(group)
        error = np.sum(abs(RBF_network_multiple_params(population, data_X).T - data_Y) ** 3, axis = -1) / len(data_Y)  # [pop size, data batch] - [data batch]

        return error                  # shape: (POP_SIZE, )

    def selection_by_RWS(self, pop:np.ndarray, fitness_values:np.ndarray) -> np.ndarray:
        if sum(fitness_values) == 0:
            mating_pool = deepcopy(pop)
            return mating_pool

        probabilities = (1 / fitness_values) / sum(1 / fitness_values)

        mating_pool = np.random.Generator(np.random.PCG64()).choice(pop, size = self.pop_size, p = probabilities)
        return mating_pool

    def crossover(self, mating_pool:np.ndarray) -> np.ndarray:
        np.random.shuffle(mating_pool)  # 打亂mating_pool順序

        index = np.arange(len(mating_pool) / 2, dtype = np.int32) * 2  # 0, 2, 4, ......
        cr = np.random.uniform(size = len(index))
        alpha = np.where(cr < self.cross_rate, np.random.uniform(size = len(index)), 1)[None].T

        # Crossover gene information
        spring1 = mating_pool[index] * alpha + mating_pool[index + 1] * (1 - alpha)
        spring2 = mating_pool[index] * (1 - alpha) + mating_pool[index + 1] * alpha

        # Assign new gene
        mating_pool[index] = spring1
        mating_pool[index + 1] = spring2

        return mating_pool

    def mutation(self, mating_pool:np.ndarray) -> np.ndarray:
        for pop in mating_pool:
            for index in range(self.gene_size):
                mr = np.random.uniform(0,1)
                if mr < self.mutation_rate:
                    for start, end, lower, upper in self.gene_limit:
                        if index >= start and index <= end:
                            pop[index] = np.random.uniform(lower,upper)
                            break

        return mating_pool

    def execute(self) -> np.ndarray:
        pop = self.initialization()
        fitness_values = self.calc_fitness_value(pop)
        gen_best_model = {}
        for gen in tqdm(range(1, self.max_gen+1), ncols=100, desc='Training'):
            temp_dict = {}
            mating_pool = self.selection_by_RWS(pop, fitness_values)
            mating_pool = self.crossover(mating_pool)
            mating_pool = self.mutation(mating_pool)
            fitness_value = self.calc_fitness_value(mating_pool)
            pop = deepcopy(mating_pool)    # replace

            best_model_index = fitness_value.argmin()
            best_model = pop[best_model_index]
            best_fitness_value = np.min(fitness_value)

            temp_dict['model'] = best_model
            temp_dict['fitness'] = best_fitness_value
            gen_best_model[gen] = temp_dict

        # get best result
        min_fitness_model = min(gen_best_model.values(), key=lambda x: x['fitness'])['model']

        return min_fitness_model

class Car:
    def __init__(self, radius:int, length:int):
        self.radius = radius
        self.length = length
        self.position_x = 0
        self.position_y = 0
        self.orientation = 90 # range: 90 ~ -270 (degrees)

    def calc_next_step(self, theta:np.ndarray):
        next_x = self.position_x + np.cos(np.radians(self.orientation + theta)) + np.sin(np.radians(theta))*np.sin(np.radians(self.orientation))
        next_y = self.position_y + np.sin(np.radians(self.orientation + theta)) - np.sin(np.radians(theta))*np.cos(np.radians(self.orientation))
        self.position_x, self.position_y = next_x, next_y

        self.orientation = self.orientation + np.degrees(np.arcsin(2*np.sin(np.radians(theta))/self.length))
        if self.orientation < -270:  # if out of range
            self.orientation += 360
        if self.orientation > 90:
            self.orientation -= 360


def save_model(data:np.ndarray, path:str='best_model.csv'):
    with open(path,'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def load_model(path:str='best_model.csv'):
    try:
        model = np.genfromtxt(path, delimiter=',')
        return model
    except FileNotFoundError:
        print(f"[Errno 2] No such file or directory: '{path}'")


def calc_distance_from_2_point(point1:tuple, point2:tuple) -> float:
    # 給兩點，求距離
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    distance = np.hypot(x2-x1,y2-y1)

    return distance

def calc_line_with_point_and_angle(point:tuple, angle:float) -> tuple[float]:
    # 給點和角度，求線
    x,y = point
    theta = np.radians(angle)
    
    if angle % 180 == 90: # 垂直
        A = 1
        B = 0
        C = -x
    else:
        m = np.tan(theta)
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
    theta = np.radians(angle)
    unit_vector_x = np.cos(theta)
    unit_vector_y = np.sin(theta)

    return unit_vector_x, unit_vector_y

def calc_dF_dL_dR(walls:list[tuple], car:Car) -> tuple[np.float64]:
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

        # 濾除範圍外的交點
        cnt = 0
        filtered_intersection = []
        for point in intersection:
            if point is not None:
                # point = (round(point[0],2), round(point[1],2))  # TODO: TEST
                if point[0] >= walls[cnt][0][0]-1e-5 and point[0] <= walls[cnt][1][0]+1e-5 and \
                point[1] >= walls[cnt][0][1]-1e-5 and point[1] <= walls[cnt][1][1]+1e-5: # 確認交點是否在線段範圍內
                    
                    filtered_intersection.append(point)
            cnt += 1

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
        if len(temp_distance) > 0:
            result_distance.append(min(30,min(temp_distance)))
        else:
            result_distance.append(float('inf'))

    dF = result_distance[0]
    dL = result_distance[1]
    dR = result_distance[2]

    return dF, dL, dR


if __name__ =='__main__':

    walls = [((-6, -10), (-6, 22)),
                ((-6, 22), (18, 22)),
                ((18, 22), (18, 50)),
                ((18, 50), (30, 50)),   # 終點邊界  原始為37 -> 50
                ((30, 10), (30, 50)),
                ((6, 10), (30, 10)),
                ((6, -10), (6, 10)),
                ((-6, -10), (6, -10))]  # 起點邊界  原始為0 -> -10
    
    # walls = [((-9, -10), (-9, 25)),     # 拓寬版本
    #             ((-9, 25), (15, 25)),
    #             ((15, 25), (15, 60)),
    #             ((15, 60), (33, 60)),
    #             ((33, 7), (33, 60)),
    #             ((9, 7), (33, 7)),
    #             ((9, -10), (9, 7)),
    #             ((-9, -10), (9, -10))]  
    
    goal_position = (24, 37)

    # 若已有模型，此段可不跑
    if TRAIN == True:
    # ===========training data 建立模型===========start
        best_model = GeneticAlgorithmRealNum(pop_size=POP_SIZE,
                                    input_dimension=INPUT_DIMENSION,
                                    node_cnt=NODE_CNT,
                                    cross_rate=CROSS_RATE,
                                    mutation_rate=MUTATION_RATE,
                                    max_gen=MAX_GEN).execute()

        save_model(best_model, 'best_model.csv')
    # ===========training data 建立模型===========end
    if TEST == True:
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
        plt.plot(car.position_x, car.position_y, marker='s', color='r', fillstyle='none')

        it = 0 # iteration
        result_distance = [100, 100, 100] # initial distance for while iterating
        while (int(car.position_x), int(car.position_y)) != goal_position and (it < 300):
            dF, dL, dR = calc_dF_dL_dR(walls, car)

            if dF == float('inf') or dL == float('inf') or dR == float('inf'):
                print('out of range!')
                plt.savefig('out_of_range.png')
                break

            theta = RBF_network_single_params(best_model, np.array([[dF, dL, dR]]))
            theta = np.where(theta < 0., 0, theta)
            theta = np.where(theta > 1., 1, theta)
            theta = output_scale(theta)[0]

            print(f'Iter: {it}, Position: ({car.position_x:.02f}, {car.position_y:.02f}), Orientation: {car.orientation:.02f}°, theta: {theta:.2f}°, FLR: {dF:.01f}, {dL:.01f}, {dR:.01f}')
            plt.plot(car.position_x, car.position_y, marker='s', color='r', fillstyle='none')
            plt.pause(0.05) # 畫面更新間隔
            
            car.calc_next_step(theta=theta)
            it += 1
            if (int(car.position_x), int(car.position_y)) == goal_position:
                plt.savefig('success.png')

        plt.show()