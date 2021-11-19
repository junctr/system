import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

x0 = np.array([
    [-4],
    [-3],
],dtype = np.float64)

x_star = np.array([
    [1],
    [1],
],dtype = np.float64)

x = np.zeros((2,1),dtype = np.float64)

a_gd_list = [0.00005, 0.0001, 0.0002]

def J(x):

    j = 100 * (x[1][0] - x[0][0]**2)**2 + (1 - x[0][0])**2

    return j

def djdx(x):
    
    djdx = np.array([
        [(-400 * x[0][0] * (x[1][0] - x[0][0]**2) - 2 * (1 - x[0][0])), (200 * (x[1][0] - x[0][0]**2))],
    ],dtype = np.float64)
    
    return djdx

def d2jdx2(x):
    
    d2jdx2 = np.array([
        [(-400 * (x[1][0] - x[0][0]**2) + 800 * x[0][0]**2 + 2), (-400 * x[0][0])],
        [(-400 * x[0][0]), (200)],
    ],dtype = np.float64)
    
    return d2jdx2

def gradient_descent(a_gd, x):
    
    p = - djdx(x).T
    
    dx = a_gd * p
    
    return dx

def newton(x):
    
    dx = -np.linalg.inv(d2jdx2(x))@djdx(x).T
    
    return dx

def main():
    
    x_j = np.linspace(-4.5, 2.5, 1000)
    y_j = np.linspace(-25, 20, 1000)
    X, Y = np.meshgrid(x_j, y_j)
    Z = 100 * (Y - X**2)**2 + (1 - X)**2
    
    x_data = [[x0[0][0]] for i in range(len(a_gd_list) + 1)]
    y_data = [[x0[1][0]] for i in range(len(a_gd_list) + 1)]
    
    i_gd = 0
        
    for a_gd in a_gd_list:
        
        x = x0.copy()
        
        print(a_gd)
        
        for i in tqdm(range(1000000)):
            
            x += gradient_descent(a_gd, x)
            
            x_data[i_gd].append(x[0][0])
            y_data[i_gd].append(x[1][0])
            
            if abs(J(x_star) - J(x)) < 0.00001:
                
                break
            
        i_gd += 1
    
    x = x0.copy()
    print("newton")
    
    for i in tqdm(range(100)):
        
        x += newton(x)
        
        x_data[i_gd].append(x[0][0])
        y_data[i_gd].append(x[1][0])
        
        if abs(J(x_star) - J(x)) < 0.00001:
                
                break
    
    fig = plt.figure()
        
    ax0 = fig.add_subplot(1,1,1)
    ax0.plot(x_data[0], y_data[0], marker=".", label="gd(0.00005)")
    ax0.plot(x_data[1], y_data[1], marker=".", label="gd(0.0001)")
    ax0.plot(x_data[2], y_data[2], marker=".", label="gd(0.0002)")
    ax0.plot(x_data[3], y_data[3], marker=".", label="Newton")
    ax0.contour(X, Y, Z, 100)
    
    ax0.set_xlabel("$\it{x_{1}}$",fontsize=15)
    ax0.set_ylabel("$\it{x_{2}}$",fontsize=15)
    
    ax0.legend()
    
    fig.savefig("fig_2.png")
        
    return



if  __name__ == "__main__":
    main()
