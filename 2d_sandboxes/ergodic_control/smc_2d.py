import numpy as np
from js import Path2D


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.x0 = [.2, .3]  # Initial position
param.nbFct = 10     # Number of basis functions along x and y
param.nbVar = 2      # Dimension of the space
param.nbGaussian = 2 # Number of Gaussians to represent the spatial distribution
param.dt = 1e-2      # Time step
param.u_max = 1e1    # Maximum speed allowed
param.xlim = [0, 1]  # Domain limit

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
x = None
t = None
wt = None


# Reset function
# ===============================
def reset():
    global x, t, wt, param, controls, paths

    # Retrieve the initial position defined by the user
    param.x0 = np.array(param.x0)
    if (len(param.x0.shape) != 1) or (param.x0.shape[0] != 2):
        print("Error: 'param.x0' must be a vector of size 2")
        return

    param.x0 = np.clip(param.x0, 0.01, 0.99) # x0 should be within [0,1]

    # Retrieve the number of gaussians defined by the user, and create/delete existing ones as needed
    param.nbGaussian = max(int(param.nbGaussian), 1)
    update_gaussians(param)

    # Compute Fourier series coefficients w_hat of desired spatial distribution
    param.rg = np.arange(0, param.nbFct, dtype=float)
    KX = np.zeros((param.nbVar, param.nbFct, param.nbFct))
    KX[0,:,:], KX[1,:,:] = np.meshgrid(param.rg, param.rg)

    sp = (param.nbVar + 1) / 2 # Sobolev norm parameter
    param.Lambda = np.array(KX[0,:].flatten()**2 + KX[1,:].flatten()**2 + 1).T**(-sp)
    param.op = hadamard_matrix(2**(param.nbVar-1))
    param.op = np.array(param.op)
    param.kk = KX.reshape(param.nbVar, param.nbFct**2) * param.omega

    alpha = np.ones(param.nbGaussian) / param.nbGaussian # mixing coeffs. Priors
    param.w_hat = fourier(alpha)

    param.xx, param.yy = np.meshgrid(np.arange(1, param.nbFct+1), np.arange(1, param.nbFct+1))

    # Reset the variables
    x = param.x0.copy()
    t = 0
    wt = np.zeros(param.nbFct**param.nbVar)
    r_x = np.array((0, 2))
    r_g = None

    controls = create_gaussian_controls(param)
    paths = [ Path2D.new() ]


# Update function
# ===============================
def update():
    global x, t, wt, param, paths

    t += 1
    x_prev = x.copy()

    # Retrieve the command
    u, wt = control(x, t, wt, param)

    # Update of the position
    x += u * param.dt

    # Update the path (for rendering)
    path = paths[0]
    path.moveTo(x_prev[0], x_prev[1])
    path.lineTo(x[0], x[1])


# Rendering function
# ===============================
def draw_histograms():
    global wt

    if wt is None:
        return

    ctx_histogram.setTransform(
        histogram_area_rect[2], 0, 0, -histogram_area_rect[3],
        histogram_area_rect[0], histogram_area_rect[3] + histogram_area_rect[1]
    )

    w_min = np.min(param.w_hat)
    w_max = np.max(param.w_hat)
    w_hat = (np.reshape(param.w_hat, [param.nbFct, param.nbFct]).T - w_min) / (w_max - w_min)

    wt2 = (np.reshape(wt / t, [param.nbFct, param.nbFct]).T - w_min) / (w_max - w_min)
    dim = 0.48 / param.nbFct

    for ky in range(param.nbFct):
        for kx in range(param.nbFct):
            color = (1.0 - w_hat[ky, kx]) * 255
            ctx_histogram.fillStyle = f'rgb({color}, {color}, {color})'
            ctx_histogram.fillRect(kx*dim, 1.0 - ky*dim*2, dim*1.1, -dim*2.2)

            color = (1.0 - wt2[ky, kx]) * 255
            ctx_histogram.fillStyle = f'rgb({color}, {color}, {color})'
            ctx_histogram.fillRect(0.52 + kx*dim, 1.0 - ky*dim*2, dim*1.1, -dim*2.2)

    left = 0.005
    right = 0.48 + dim * 0.1 - 0.005
    top = 1.0
    bottom = 1.0 - (0.48 * 2 + dim * 0.2)

    ctx_histogram.strokeStyle = 'rgb(255, 165, 0)'

    ctx_histogram.lineWidth = 0.01
    ctx_histogram.beginPath()
    ctx_histogram.moveTo(left, top)
    ctx_histogram.lineTo(left, bottom)
    ctx_histogram.moveTo(right, top)
    ctx_histogram.lineTo(right, bottom)
    ctx_histogram.stroke()

    top = 0.99
    bottom = 1.0 - (0.48 * 2 + dim * 0.2) + 0.01

    ctx_histogram.lineWidth = 0.02
    ctx_histogram.beginPath()
    ctx_histogram.moveTo(left, top)
    ctx_histogram.lineTo(right, top)
    ctx_histogram.moveTo(left, bottom)
    ctx_histogram.lineTo(right, bottom)
    ctx_histogram.stroke()

    element = document.getElementsByClassName('legend-left')[0]
    element.style.paddingLeft = f'{histogram_area_rect[0]}px'

    element = document.getElementsByClassName('legend-right')[0]
    element.style.paddingRight = f'{histogram_area_rect[0]}px'
