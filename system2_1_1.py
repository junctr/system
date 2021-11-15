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

a_gd_list = [0.01, 0.1, 0.5]

def J(x):

    j = (1/2)*x.T@Q@x + b.T@x

    #j = 3/2*x[0][0]**2 + 1/2*x[1][0]**2 - x[0][0]*x[1][0] - 2*x[0][0]

    return j[0][0]

def djdx(x):
    
    djdx = np.array([
        [(3*x[0][0] - x[1][0] -2), (x[1][0] - x[0][0])],
    ],dtype = np.float64)
    
    return djdx

def d2jdx2(x):
    
    d2jdx2 = np.array([
        [3, -1],
        [-1, 1],
    ],dtype = np.float64)
    
    return d2jdx2

def gradient_descent(a_gd, x):
    
    p = - djdx(x).T
    
    dx = a_gd * p
    
    return dx

def newton(x):
    
    dx = -np.linalg.inv(d2jdx2(x))@djdx(x).T
    
    return dx

# def main():
#     for i in tqdm(range(100000)):

#         time.sleep(0.0001)



def main():
    
    x_j = np.linspace(-5, 5, 1000)
    y_j = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x_j, y_j)
    Z = 3/2*X**2 + 1/2*Y**2 - X*Y - 2*X
    
    #x_data = [x0[0][0]]
    #y_data = [x0[1][0]]
    
    # x_data = [[x0[0][0]] for i in range(len(a_gd_list))]
    # y_data = [[x0[1][0]] for i in range(len(a_gd_list))]
    
    x_data = [[x0[0][0]] for i in range(len(a_gd_list) + 1)]
    y_data = [[x0[1][0]] for i in range(len(a_gd_list) + 1)]
    
    
    
    #x = x0.copy()
    
    # for i in tqdm(range(100)):
        
    #     x += gradient_descent(a_gd, x)
        
    #     x_data.append(x[0][0])
    #     y_data.append(x[1][0])
        
    #     print(x)
    
    i_gd = 0
        
    for a_gd in a_gd_list:
        
        x = x0.copy()
        
        print(a_gd)
        print(x)
        
        for i in tqdm(range(1000)):
            
            x += gradient_descent(a_gd, x)
            
            x_data[i_gd].append(x[0][0])
            y_data[i_gd].append(x[1][0])
            
            #print(x)
            
        i_gd += 1
    
    x = x0.copy()
    print("newton")
    print(x)
    for i in tqdm(range(100)):
        
        x += newton(x)
        
        x_data[i_gd].append(x[0][0])
        y_data[i_gd].append(x[1][0])
        
        #print(x)
    
    fig = plt.figure()
    # ax1  = fig.add_subplot(3,1,1)
    # ax2  = fig.add_subplot(3,1,2)
    # ax3  = fig.add_subplot(3,1,3)
    
    # ax1.plot(x_data[0], y_data[0])
    # ax2.plot(x_data[1], y_data[1])
    # ax3.plot(x_data[2], y_data[2])
    
    # ax1.contour(X, Y, Z, 100)
    # ax2.contour(X, Y, Z, 100)
    # ax3.contour(X, Y, Z, 100)
    
    ax0 = fig.add_subplot(1,1,1)
    ax0.plot(x_data[0], y_data[0])
    ax0.plot(x_data[1], y_data[1])
    ax0.plot(x_data[2], y_data[2])
    ax0.plot(x_data[3], y_data[3])
    ax0.contour(X, Y, Z, 100)
    
    fig.savefig("fig_gd.png")
    
    

    return



if  __name__ == "__main__":
    main()
