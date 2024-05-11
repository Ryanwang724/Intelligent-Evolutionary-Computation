import matplotlib.pyplot as plt
import numpy as np
import os
from simplex import simplex

def eggholder(x, y):
    return -(y+47)*np.sin(np.sqrt(abs(x/2+(y+47)))) - x*np.sin(np.sqrt(abs(x-(y+47))))

def plot_value():
    ax.contourf(X1, X2, all_value, levels=500)
    ax.set_title(f'Simplex Search eggholder function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

def evaluate_object_value(points):
    return eggholder(points[:, 0], points[:, 1])

def new_simplex(p1:tuple,p2:tuple,p3:tuple):
    temp_dict = {}
    temp_dict[p1] = eggholder(p1[0], p1[1])
    temp_dict[p2] = eggholder(p2[0], p2[1])
    temp_dict[p3] = eggholder(p3[0], p3[1])
    return temp_dict

LOWER = -1000
UPPER = 1000
MAXGEN = 15
x1 = np.linspace(LOWER, UPPER, 100)
x2 = np.linspace(LOWER, UPPER, 100)
X1, X2 = np.meshgrid(x1,x2)
all_value = eggholder(X1,X2)


fig, ax = plt.subplots()
plt.ion()   # 開啟 interactive mode

plot_value()
points = fig.ginput(3)   # 讓使用者點3個點
print(f'points: {points}')
points = np.array(points)
plot_value()
ax.plot(points[:, 0], points[:, 1], marker='o', color='red') 
ax.plot(points[[0, 1, 2, 0], 0], points[[0, 1, 2, 0], 1], color='blue')
plt.show()

obj_value = evaluate_object_value(points)
print(f'object_value: {obj_value}')

obj_dict = {}
for point in points:
    obj_dict[tuple(point)] = eggholder(point[0], point[1])
print(f'object_dict: {obj_dict}')

simp = simplex()
for gen in range(0,MAXGEN):
    simp.relationship(obj_dict)
    simp.calc_m()
    simp.reflection()
    if eggholder(simp.r[0],simp.r[1]) < eggholder(simp.b[0],simp.b[1]):
        simp.expansion()
        if eggholder(simp.e[0],simp.e[1]) < eggholder(simp.b[0],simp.b[1]):
            obj_dict = new_simplex(simp.b, simp.g, simp.e)
        else:
            obj_dict = new_simplex(simp.b, simp.g, simp.r)
    elif eggholder(simp.b[0],simp.b[1]) < eggholder(simp.r[0],simp.r[1]) and eggholder(simp.r[0],simp.r[1]) < eggholder(simp.g[0],simp.g[1]):
        obj_dict = new_simplex(simp.b, simp.g, simp.r)
    elif eggholder(simp.g[0],simp.g[1]) < eggholder(simp.r[0],simp.r[1]):
        simp.contraction()
        if eggholder(simp.c[0],simp.c[1]) < eggholder(simp.w[0],simp.w[1]):
            obj_dict = new_simplex(simp.b, simp.g, simp.c)
        else:
            simp.shrink()
            obj_dict = new_simplex(simp.b, simp.g, simp.s)

    points = [simp.b, simp.g, simp.w]
    points = np.array(points)
    plot_value()
    ax.plot(points[:, 0], points[:, 1], marker='o', color='red') 
    ax.plot(points[[0, 1, 2, 0], 0], points[[0, 1, 2, 0], 1], color='blue')
    plt.pause(0.5)
    plt.show()

if not os.path.exists(f'./img/eggholder'):
    os.makedirs(f'./img/eggholder')

plt.savefig(f'./img/eggholder/eggholder_{MAXGEN}.png')