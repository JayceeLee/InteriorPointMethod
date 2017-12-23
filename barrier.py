import numpy as np
import scipy as sp
from scipy import optimize
from scipy import linalg

def SGD(params, lr):
    for param in params:
        param = param - lr * param.grad

class Problem(object):
    """docstring for Problem."""
    def __init__(self, obj=None, A=None, b=None, F=None, g=None, 
                    eps=1e-8, t=1, mu=1.1):
        super(Problem, self).__init__()
        self.eps = eps
        self.t = t
        self.mu = mu
        assert obj is not None

        self.obj = obj
        # inequality constraints
        self.F = F
        self.g = g
        # equality constraints
        self.A = A
        self.b = b

        # shape of A
        self.m, self.n = A.shape

class Barrier(object):
    """docstring for IPM."""
    def __init__(self, prob, arg=None):
        super(Barrier, self).__init__()
        self.prob = prob
        self.iter = 0
        self.totalTime = 0.0
        self.residual = 0.0
        self.x = np.zeros(self.prob.b.shape).astype(float)
        self.y = np.zeros(self.prob.b.shape).astype(float)
    def initialize_start_point(self):
        """
        Barrier method requires starting from strictly feasible point.
        For LP and QP (for example):
            Ax = b
            Fx < g
            strictly inequality is required.
        Goal: This function is to start from such a strictly feasible point.
        Return: a vector x that is strictly feasible.
        """

        # first: find feasible point under equality constraint: Ax = b
        # use normal equation first
        # i.e. x = inverse(A.T * A) * A.T * b
        # TODO: ---
        hat_inv = self.prob.A.T.dot(self.prob.A).I
        self.x = hat_inv.dot(self.prob.A.T).dot(self.prob.b)
        
        # second: move x to strictly feasible point
        # Fx < g
        coeff_matrix = self.prob.A.dot(self.prob.A.T)
        inv_coeff = coeff_matrix.I
        self.y = inv_coeff.dot(self.prob.A).dot(self.prob.b)
        self.s = self.prob.b - self.prob.A.T.dot(self.y) + self.prob.F.dot(self.x)

        deltax = -1.5 * self.x.min() if -1.5 * self.x.min() > 0 else 0
        deltas = -1.5 * self.s.min() if -1.5 * self.s.min() > 0 else 0

        unitvec = np.ones(self.x.shape)
        self.x = self.x + deltax * unitvec
        self.s = self.s + deltas * unitvec
        
    def centering(self):
        def cons():
            return np.sum(0)
        constraints = {
            'type': 'eq',
            'fun': np.sum(np.dot(self.prob.A, self.x) - self.prob.b)
        }
            
        
        opt_result = sp.optimize.minimize(self.prob.obj, self.x,
                                    constraints=constraints, method='BFGS')
        return opt_result.x

    def check_criterion(self):
        return self.prob.m / self.prob.t < self.prob.eps

    def increase_t(self):
        self.prob.t = self.prob.mu * self.prob.t

    def optimize(self):
        # initialze
        self.initialize_start_point()

        # repeat until converge
        while True:
            # centering and update
            self.x = self.centering()
            break
            # check stop
            if self.check_criterion():
                break
            self.increase_t()
    def print_result(self):
        print(self.x)
        
if __name__ == '__main__':
    def obj(x):
        return np.linalg.norm(np.dot(x,x))
    A = np.mat([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([4.0, 6.0]).reshape((-1, 1)).astype(float)
    F = np.array([[1.0, 0.0], [0.0, 1.0]])
    g = np.array([2.0, 2.0]).reshape((-1, 1)).astype(float)
    prob = Problem(obj=obj, A=A, b=b, F=F, g=g)
    solver = Barrier(prob=prob)
    solver.optimize()
    solver.print_result()

