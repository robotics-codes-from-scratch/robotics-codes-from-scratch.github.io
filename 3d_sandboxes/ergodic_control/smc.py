import numpy as np


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.nbFct = 10     # Number of basis functions along x, y and z
param.nbVar = 3      # Dimension of the space
param.nbGaussian = 2 # Number of Gaussians to represent the spatial distribution
param.dt = 1e-2      # Time step
param.u_max = 10.0    # Maximum speed allowed
param.xlim = [0, 1]  # Domain limit

# Initial robot state
param.x0 = [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 1.0]

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
t = None
wt = None
trajectory = None


# Reset function
# ===============================
def reset(reset_state=True):
    global t, wt, param, trajectory

    # Retrieve the initial state defined by the user
    param.x0 = np.array(param.x0)
    if (len(param.x0.shape) != 1) or (param.x0.shape[0] != 7):
        print("Error: 'param.x0' must be a vector of size 7")
        return

    robot.jointPositions = param.x0

    # Retrieve the number of gaussians defined by the user, and create/delete existing ones as needed
    param.nbGaussian = max(int(param.nbGaussian), 1)
    update_gaussians(param)

    # Compute the desired spatial distribution
    param.rg = np.arange(0, param.nbFct, dtype=float)
    param.kk = np.ndarray((param.nbVar, param.nbFct**param.nbVar))

    for n in range(param.nbVar):
        param.kk[n,:] = np.tile(np.repeat(param.rg, param.nbFct**n), param.nbFct**(param.nbVar-n-1))

    sp = (param.nbVar + 1) / 2 # Sobolev norm parameter
    param.Lambda = (np.sum(param.kk**2, axis=0) + 1).T**(-sp)

    param.kk *= param.omega

    param.op = hadamard_matrix(2**(param.nbVar-1))
    param.op = np.array(param.op[:3,:])

    alpha = np.ones(param.nbGaussian) / param.nbGaussian # mixing coeffs. Priors
    param.w_hat = fourier(alpha)

    param.yy, param.zz, param.xx = np.meshgrid(np.arange(1,param.nbFct+1), np.arange(1,param.nbFct+1), np.arange(1,param.nbFct+1))

    # Reset the variables
    t = 0
    wt = np.zeros(param.nbFct**param.nbVar)

    trajectory = np.ndarray((1, 3))
    trajectory[0,:] = robot.fkin(robot.jointPositions)[:3]

    reset_rendering(param)


# Update function
# ===============================
def update():
    global t, wt, trajectory, param

    t += 1

    # Retrieve the command
    u, wt = controlCommand(robot.jointPositions, t, wt, param)

    # Apply the command to the robot
    robot.control = robot.control + u * 0.2

    # Update the list of points used to draw the trajectory
    ee = robot.fkin(robot.jointPositions)
    if (trajectory.shape[0] < 3) or (np.linalg.norm(trajectory[-2,:] - ee[:3]) >= 0.05):
        trajectory = np.append(trajectory, [ee[:3]], axis=0)
    else:
        trajectory[-1,:] = ee[:3]
