import copy
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

class finite_difference_methods:
    def __init__(self, time_steps, state_steps, time_min, time_max, state_min, state_max):
        self.time_steps = time_steps
        self.state_steps = state_steps
        self.time_min = time_min
        self.time_max = time_max
        self.state_min = state_min
        self.state_max = state_max
        self.grid = np.zeros(
            shape = (state_steps+1, time_steps+1)
        )
        self.dx = (state_max-state_min)/state_steps
        self.dt = (time_max - time_min)/time_steps
        self.x_grid = np.linspace(start = state_min,
                                  stop = state_max,
                                  num = state_steps+1)
        self.t_grid = np.linspace(
            start = time_min,
            stop = time_max,
            num = time_steps+1
        )
        self.lambda_ = self.dt / self.dx ** 2
        self.grid[:, 0] = self.initial_cond(self.x_grid, 0)
        self.grid[0, :] = 0
        self.grid[-1, :] = 0
        self.calculated = None

    def thomas_algorithm(self, a, b, c, d):
        # Solves a tridiagonal system of equations, i.e Ax = b, where A is tridiagonal
        dim = len(d)  # nr of equations to be solved
        ac, bc, cc, dc = map(np.array, (a, b, c, d))
        for i in range(1, dim):
            w = ac[i - 1] / bc[i - 1]
            bc[i] = bc[i] - w * cc[i - 1]
            dc[i] = dc[i] - w * dc[i - 1]
        x = bc
        x[-1] = dc[-1] / bc[-1]

        for j in range(dim - 2, -1, -1):
            x[j] = (dc[j] - cc[j] * x[j + 1]) / bc[j]
        return x

    def tridiag(self, a, b, c, k1=-1, k2=0, k3=1):
        return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

    def A_matrix(self, a_, b_, c_, dim):

        a = np.repeat(a_, dim-1)
        b = np.repeat(b_, dim)
        c = np.repeat(c_, dim-1)
        mat = self.tridiag(a, b, c)
        return mat

    def initial_cond(self, x, t=0):
        return np.exp(-(math.pi**2) * t) * np.sin(math.pi * x)

    def plot_with_true(self, t):
        if self.calculated is None:
            raise Exception('The class has not calculated any results yes.')
        plt.plot(self.x_grid, self.calculated[str(t)], 'r-')
        temp = np.exp(-(math.pi**2) * t) * np.sin(math.pi * self.x_grid)
        plt.plot(self.x_grid, temp, 'b')
        plt.show()

    def surface_plot(self):
        if self.calculated is None:
            raise Exception('Use one of the methods to solve the grid before plotting.')
        fig = plt.figure(figsize=(40, 6))
        ax2 = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(self.t_grid, self.x_grid)
        ax2.plot_surface(Y, X, self.calculated, cmap=cm.ocean)
        ax2.set_title("BS price surface")
        ax2.set_xlabel("S")
        ax2.set_ylabel("t")
        ax2.set_zlabel("V")
        ax2.view_init(30, 40)  # this function rotates the 3d plot
        plt.show()

    def explicit_method(self):
        if self.lambda_ > 0.5:
            raise Exception('The explicit method is not stable in this case. Increase dt of decrease dx')

        res = copy.deepcopy(self.grid)
        A = self.A_matrix(
            self.lambda_,
            1-2 * self.lambda_,
            self.lambda_,
            self.state_steps - 1
        )
        offset = np.zeros(self.state_steps - 1)
        for i in range(self.time_steps):
            offset[0] = res[0, i] * self.lambda_
            offset[-1] = res[-1, i] * self.lambda_
            res[1:-1, i+1] = np.dot(A, res[1:-1, i]) + offset

        c_names = [str(self.time_min + self.dt * i) for i in range(self.time_steps + 1)]
        df = pd.DataFrame(res, columns=c_names)

        self.calculated = df
        return res

    def implicit_method(self):
        res = copy.deepcopy(self.grid)
        A = self.A_matrix(
            -self.lambda_,
            2 * self.lambda_ + 1,
            -self.lambda_,
            self.state_steps - 1
        )
        offset = np.zeros(self.state_steps - 1)
        for i in range(self.time_steps):
            offset[0] = res[0, i+1]* (-self.lambda_)
            offset[-1] = res[-1, i+1] * (-self.lambda_)
            RHS = res[1:-1, i] + offset
            res[1:-1, i + 1] = self.thomas_algorithm(
                np.diag(A, -1),
                np.diag(A),
                np.diag(A, +1),
                RHS
            )
        c_names = [str(self.time_min + self.dt * i) for i in range(self.time_steps + 1)]
        df = pd.DataFrame(res, columns=c_names)
        self.calculated = df
        return res

    def crank_nicolson(self):
        res = copy.deepcopy(self.grid)
        A = self.A_matrix(
            -self.lambda_/2,
            self.lambda_ + 1,
            -self.lambda_/2,
            self.state_steps - 1
        )
        B = self.A_matrix(
            self.lambda_/2,
            1 - self.lambda_,
            self.lambda_/2,
            self.state_steps - 1
        )
        offset = np.zeros(self.state_steps - 1)
        for i in range(self.time_steps):
            offset[0] = res[0, i] * (self.lambda_/2)
            offset[-1] = res[-1, i + 1] * (self.lambda_/2)
            bw = np.dot(B, res[1:-1, i] + offset) # The right hand side

            offset[0] = res[0, i + 1] * (-self.lambda_/2)
            offset[-1] = res[-1, i + 1] * (-self.lambda_/2)
            RHS = bw + offset
            res[1:-1, i + 1] = self.thomas_algorithm(
                np.diag(A, -1),
                np.diag(A),
                np.diag(A, +1),
                RHS
            )
        c_names = [str(self.time_min + self.dt * i) for i in range(self.time_steps + 1)]
        df = pd.DataFrame(res, columns=c_names)
        self.calculated = df
        return res

'''
Example of usage with a known solution to a PDE given as
y(x,t) = exp(-(pi**2 * t) * sin( pi * x)
The inputs in the class are 

time_steps: int. 
state_steps: int.
time_min: float. 
time_max: float. 
state_max: float. 
state_min: float.

Beware when using the explicit method, the solution can be unstable, if 
\delta t/\delta x^(2) > 1, and 'explodes'.

Example use of the class '''
finite_difference_solutions = finite_difference_methods(
    200,
    10,
    0,
    1,
    0,
    1
)
finite_difference_solutions.explicit_method()
finite_difference_solutions.plot_with_true(1.0)
finite_difference_solutions.surface_plot()


