import numpy as np
from autograd import jacobian

# x = np.array([[1],[2]],dtype=float)
# x = np.array([1,2],dtype=float)
x = [1.0,2.0]
def f(x):
    return x[0]**2 + 2 * x[1]**2

print(f(x))

# j = jacobian(f(x),x)
j = jacobian(f,x)

print(j)