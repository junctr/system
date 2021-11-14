import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

Q = np.array([
    [3, -1],
    [-1, 1],
],dtype = np.float64)

b = np.array([
    [-2],
    [0],
],dtype = np.float64)

x0 = np.array([
    [-4],
    [-3],
],dtype = np.float64)

x_star = np.array([
    [1],
    [1],
],dtype = np.float64)

x = np.zeros((2,1),dtype = np.float64)

a_gd = [0.01, 0.1, 0.5]

def J(x):

    j = (1/2)*x.T@Q@x + b.T@x

    #j = 3/2*x[0][0]**2 + 1/2*x[1][0]**2 - x[0][0]*x[1][0] - 2*x[0][0]

    return j[0][0]

def djdx(x):
    
    djdx = np.array([
        [(3*x[0][0] - x[1][0] -2), (x[1][0] - x[0][0])],
    ])
    return djdx

def gradient_descent(a_gd, x):
    
    p = - djdx(x).T
    
    dx = a_gd * p
    
    return dx

# def main():
#     for i in tqdm(range(100000)):

#         time.sleep(0.0001)

def main():
    
    x_j = np.linspace(-5, 5, 1000)
    y_j = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x_j, y_j)
    Z = 3/2*X**2 + 1/2*Y**2 - X*Y - 2*X
    
    x_data = [x0[0][0]]
    y_data = [x0[1][0]]
    
    x = x0.copy()
    
    for i in tqdm(range(100)):
        
        x += gradient_descent(a_gd, x)
        
        x_data.append(x[0][0])
        y_data.append(x[1][0])
        
        print(x)
        
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    
    ax.plot(x_data, y_data)
    ax.contour(X, Y, Z, 100)
    
    fig.savefig("fig_gd.png")
        
    

    return



if  __name__ == "__main__":
    main()
