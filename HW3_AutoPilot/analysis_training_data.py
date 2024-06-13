import numpy as np
import os
import matplotlib.pyplot as plt


theta_scale = lambda theta: (theta + 40) / 80.0                       # [-40, 40] -> [0, 1] for read training data
def load_data() -> tuple[np.ndarray]:
    training_data_path = 'HW3_training_data'
    file_list = [_ for _ in os.listdir(training_data_path) if _.endswith(r'.txt') and _[:6] == "train_"] # 取出符合條件的檔案

    X, Y =[], []
    for file in file_list:
        with open(f'{training_data_path}/{file}', 'r') as f:
            content = [list(map(float, line[:-1].split(' '))) for line in f if '30.0000000' not in line]

        X += [list(map(float, data[:3])) for data in content]
        Y += [theta_scale(float(data[-1])) for data in content]

    return np.array(X), np.array(Y)


if __name__ == '__main__':
    data_X, data_Y = load_data()

    plt.figure()
    plt.hist(data_X[:, 0], bins = 30, density = True, alpha = 0.6, color = 'g')
    plt.title('forward')
    plt.savefig('img/analysis_training_data/dF.png')

    plt.figure()
    plt.hist(data_X[:, 1], bins = 30, density = True, alpha = 0.6, color = 'g')
    plt.title('left')
    plt.savefig('img/analysis_training_data/dL.png')

    plt.figure()
    plt.hist(data_X[:, 2], bins = 30, density = True, alpha = 0.6, color = 'g')
    plt.title('right')
    plt.savefig('img/analysis_training_data/dR.png')
    plt.show()