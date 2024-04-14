import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def rosenbrock(x1, x2):
    return 100*(x2-x1**2)**2 + (1-x1)**2

def eggholder(x, y):
    return -(y+47)*np.sin(np.sqrt(abs(x/2+(y+47)))) - x*np.sin(np.sqrt(abs(x-(y+47))))

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def rastrigin(x1, x2):
    return 10*2 + (x1**2 - 10*np.cos(2*np.pi*x1)) + (x2**2 - 10*np.cos(2*np.pi*x2))





# ga = GeneticAlgorithm(pop_size, gene_size, cross_rate, mutation_rate=, func=rosenbrock)
# ga.execute()


x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1,x2)
result = rosenbrock(X1,X2)

plt.contourf(X1,X2,result, locator=ticker.LogLocator(),alpha = 0.5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar()
plt.title('GA Rosenbrock function')
plt.show()