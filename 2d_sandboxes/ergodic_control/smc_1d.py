import numpy as np
from js import Path2D


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.x0 = .1        # Initial position
param.nbFct = 10     # Number of basis functions
param.nbGaussian = 1 # Number of Gaussians to represent the spatial distribution
param.dt = 1e-2      # Time step
param.u_max = 2.0   # Maximum speed allowed
param.xlim = [0, 1]  # Domain limit

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
x = 0
t = 0
r_x = []
wt = np.zeros((param.nbFct, 1))

path_1d = None
paths_2d = None


# Reset function
# ===============================
def reset():
    global x, r_x, t, wt, param, path_1d, paths_2d

    # Retrieve the initial state defined by the user
    (param.x0, Mu, Sigma) = initialState()

    param.Mu = np.array(Mu)
    param.Sigma = np.array(Sigma)

    if len(param.Mu.shape) != 1:
        print("Error: 'Mu' must be a vector of 'N' values, with 'N' the number of gaussians")
        return

    param.nbGaussian = param.Mu.shape[0]

    if (len(param.Sigma.shape) != 1) or (param.Sigma.shape[0] != param.nbGaussian):
        print(f"Error: 'Sigma' must be a vector of {param.nbGaussian} values")
        return

    # Compute Fourier series coefficients w_hat of desired spatial distribution
    param.Priors = np.ones(param.nbGaussian) / param.nbGaussian # Mixing coefficients

    rg = np.arange(param.nbFct, dtype=float).reshape((param.nbFct, 1))
    param.kk = rg * param.omega
    param.Lambda = (rg**2 + 1) ** -1 # Weighting vector

    # Explicit description of w_hat by exploiting the Fourier transform
    # properties of Gaussians (optimized version by exploiting symmetries)
    param.w_hat = np.zeros((param.nbFct, 1))
    for j in range(param.nbGaussian):
        param.w_hat = param.w_hat + param.Priors[j] * np.cos(param.kk * Mu[j]) * np.exp(-.5 * param.kk**2 * param.Sigma[j])

    param.w_hat = param.w_hat / param.L

    # Reset the variables
    x = param.x0
    t = 0
    r_x = []
    wt = np.zeros((param.nbFct, 1))

    path_1d = Path2D.new()
    paths_2d = []


# Update function
# ===============================
def update():
    global x, r_x, t, wt, param, path_1d, paths_2d

    t += 1
    x_prev = x

    # Retrieve the command
    u, wt = controlCommand(x, t, wt, param)
    if isinstance(u, np.ndarray):
        u = u.flatten()[0]

    # Ensure that we don't go out of limits
    next_x = x + u * param.dt
    if (next_x < param.xlim[0]) or (next_x > param.xlim[1]):
        u = -u

    # Update of the position
    x += u * param.dt

    # Update the paths (for rendering)
    offset = (t - 1) % SPLITS_LENGTH
    if offset == 0:
        path_1d = Path2D.new()
        if len(r_x) > 0:
            path_1d.moveTo(np.min(r_x), 0.0)
            path_1d.lineTo(np.max(r_x), 0.0)

        paths_2d.append(Path2D.new())
        if len(paths_2d) > 5:
            paths_2d = paths_2d[1:]

    path_1d.moveTo(x_prev, 0.0)
    path_1d.lineTo(x, 0.0)

    paths_2d[-1].moveTo(x_prev, (offset-1) / NB_DATA_MAX * PATH_2D_HEIGHT)
    paths_2d[-1].lineTo(x, offset / NB_DATA_MAX * PATH_2D_HEIGHT)

    r_x.append(x)
