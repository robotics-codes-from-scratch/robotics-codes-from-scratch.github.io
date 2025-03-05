
import numpy as np
from math import factorial

def binomial(n, i):
    if n >= 0 and i >= 0:
        b = factorial(n) / (factorial(i) * factorial(n - i))
    else:
        b = 0
    return b

def block_diag(A, B):
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
    out[: A.shape[0], : A.shape[1]] = A
    out[A.shape[0] :, A.shape[1] :] = B
    return out

class BezierPolycurve(object):
    """Class to compute datapoints from bernstein basis functions: a quadratic or cubic concatenated bezier spline (polycurve)

    Attributes
    ----------

    nbFct: int
        Number of Bernstein basis functions for each dimension. Eiter 3 of 4 for quadratic or cubic type.
    """

    def __init__(self, nbFct):
        self.nbFct = nbFct
        self.nbSeg = 1 # Number of segments for each dimension
        self.nbIn = 1  # Dimension of input data (here: time)
        self.nbOut = 2  # Dimension of output data (here: xy)

        # Initialize weights
        if self.nbFct == 3:
            self.w = np.array([[-50.0, 350.0], [200.0, -50.0], [150.0, 500.0]])
        elif self.nbFct == 4:
            self.w = np.array([[-150.0, 250.0], [80.0, 220.0], [250.0, -50.0], [200.0, 550.0]])
        else:
            raise Exception("nbFct should be either quadratic (=3) or cubic (=4)")

        self.setup()

    def computePsi(self, t):
        """
        From a vector t with values in the range [0, 1], and matrices B and C, compute the concatenated basis functions.
        Inspired from https://gitlab.idiap.ch/rli/robotics-codes-from-scratch/-/blob/master/python/spline1D_SDF.py
        """
        T = np.zeros((1, self.nbFct))
        dT = np.zeros((1, self.nbFct))
        phi = np.zeros((len(t), self.BC.shape[1]))
        dphi = np.zeros_like(phi)
        # Compute Psi for each point
        for k in range(0, len(t)):
            # Compute residual within the segment in which the point falls
            tt = np.mod(t[k], 1 / self.nbSeg) * self.nbSeg
            # Determine in which segment the point falls in order to evaluate the basis function accordingly
            id = np.round(t[k] * self.nbSeg - tt)
            # Handle inputs beyond lower bound
            if id < 0:
                tt = tt + id
                id = 0
            # Handle inputs beyond upper bound
            if id > (self.nbSeg - 1):
                tt = tt + id - (self.nbSeg - 1)
                id = self.nbSeg - 1

            # Evaluate polynomials
            p1 = np.linspace(0, self.nbFct - 1, self.nbFct)
            p2 = np.linspace(0, self.nbFct - 2, self.nbFct - 1)
            T[0, :] = tt**p1
            dT[0, 1:] = p1[1:] * tt**p2 * self.nbSeg
            idl = id * self.nbFct + p1
            idl = idl.astype("int")

            phi[k, :] = T @ self.BC[idl, :]
            dphi[k, :] = dT @ self.BC[idl, :]

        Psi = np.kron(phi, np.eye(self.nbOut))
        dPsi = np.kron(dphi, np.eye(self.nbOut))
        return Psi, dPsi

    def setup(self):
        """
        Bézier curve in matrix form x = T * B * C * w (with phi = T * B * C). For 2d, kronecker product is used to
        obtain Psi for x = Psi * w. (cf section 6.3 of RCFS). The matrice C is not exactly the same as in the
        documentation, as a delta w is used to define intermediate points.
        """
        B0 = np.zeros((self.nbFct, self.nbFct))
        for n in range(1, self.nbFct + 1):
            for i in range(1, self.nbFct + 1):
                B0[self.nbFct - i, n - 1] = (
                    (-1) ** (self.nbFct - i - n)
                    * (-binomial(self.nbFct - 1, i - 1))
                    * binomial(self.nbFct - 1 - (i - 1), self.nbFct - 1 - (n - 1) - (i - 1))
                )
        self.B = np.kron(np.eye(self.nbSeg), B0)

        if self.nbFct == 3:
            # w1=w2-w12, w3=w2, w4=-w1+2*w2=-(w2-w12)+2*w2 => w4 =-w12+w2 , etc...
            self.C = np.zeros((self.nbFct * self.nbSeg, self.nbFct + (self.nbSeg - 1)* (self.nbFct - 2)))
            self.C[0:3, 0:3] = np.array([
                [1, 0, 0], # w0
                [0,-1, 1], # w12
                [0, 0, 1]  # w2
            ])
            for ns in range(1, self.nbSeg):
                row = ns * self.nbFct
                col = ns * (self.nbFct -2)
                self.C[row:row + self.nbFct, col: col + self.nbFct] = np.array([
                    [0, 1, 0], # w3, w6, ...
                    [0, 2, 0], # w4, w7, ...
                    [0, 0, 1]  # w5, w8, ...
                ])
                self.C[row + 1, :] -= self.C[row - 2, :] # w4, w7, ...
        else:
            # w2=w3 - delta w23,  w4=w3, w5=w3 + delta w23, etc...
            self.C = np.zeros((self.nbFct * self.nbSeg, self.nbFct + (self.nbSeg - 1)* (self.nbFct - 2)))
            self.C[0:4, 0:4] = np.array([
                [1, 0, 0, 0], # w0
                [0, 1, 0, 0], # w1
                [0, 0,-1, 1], # w23
                [0, 0, 0, 1]  # w3
            ])
            for ns in range(1, self.nbSeg):
                row = ns * self.nbFct
                col = ns * (self.nbFct -2)
                self.C[row:row + self.nbFct, col: col + self.nbFct] = np.array([
                    [0, 1, 0, 0], # w4, w8, ...
                    [1, 1, 0, 0], # w5, w9, ...
                    [0, 0,-1, 1], # w67, w10-11
                    [0, 0, 0, 1]  # w7, w11, ...
                ])

        self.BC = self.B @ self.C
        self.nbDim = 20 * self.nbSeg # Number of datapoints in a trajectory
        t = np.linspace(0, 1, self.nbDim)
        self.Psi, _ = self.computePsi(t)
        self.update_ctr_pts()

    @property
    def x(self):
        """Compute datapoints."""
        points = self.Psi @ self.w.reshape((-1, 1))
        return points

    def update_ctr_pts(self):
        """Update the (extended) control points from the weights."""
        self.ctr_pts = np.kron(self.C, np.eye(self.nbOut)) @ self.w.reshape((-1, 1))
        self.ctr_pts = self.ctr_pts.reshape((-1, 2))

    def add_segment(self, point):
        """Add a new quadratic or cubic segment in the polycurve and recompute the matrices Psi."""
        if self.nbFct == 3:
            self.w = np.vstack([self.w, point])
        else:
            self.w = np.vstack([self.w, np.zeros((2,)), point])
        self.nbSeg += 1
        self.setup()
