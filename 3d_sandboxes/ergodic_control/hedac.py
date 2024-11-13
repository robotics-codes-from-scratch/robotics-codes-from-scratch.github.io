import numpy as np
from rcfs import displayError


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.nbFct = 10     # Number of basis functions along x, y and z
param.nbVar = 3      # Dimension of the space
param.nbGaussian = 2 # Number of Gaussians to represent the spatial distribution
param.xlim = [0, 1]  # Domain limit

param.diffusion = 1  # increases global behavior
param.source_strength = 1  # increases local behavior
param.max_dx = 1.0 # maximum velocity of the agent
param.max_ddx = 0.5 # maximum acceleration of the agent
param.dx = 1
param.nbRes = 25 # resolution of discretization
param.min_kernel_val = 1e-8  # upper bound on the minimum value of the kernel
param.agent_radius = 1  # changes the effect of the agent on the coverage

# Timesteps for integrating diffusion (higher values lead to more global exploration, [1,25])
param.nb_diffusion_timesteps = 25 

param.dt = 1e-2      # Time step

# Initial robot state
param.x0 = [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 1.0]

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
agent = None
goal_density = None
coverage_density = None
heat = None
coverage_block = None
trajectory = None


# HEDAC-related functions
# ===============================
class SecondOrderAgent:
    """
    A point mass agent with second order dynamics.
    """
    def __init__(
        self,
        nbVar,
        max_dx=1,
        max_ddx=0.2,
    ):
        self.dx = np.zeros(nbVar)  # velocity

        self.t = 0  # time
        self.dt = 1  # time step

        self.max_dx = max_dx
        self.max_ddx = max_ddx

    def update(self, x, gradient):
        """
        set the acceleration of the agent to clamped gradient
        compute the position at t+1 based on clamped acceleration
        and velocity
        """
        ddx = gradient # we use gradient of the potential field as acceleration
        # clamp acceleration if needed
        if np.linalg.norm(ddx) > self.max_ddx:
            ddx = self.max_ddx * ddx / np.linalg.norm(ddx)

        x = x + self.dt * self.dx + 0.5 * self.dt * self.dt * ddx
        self.t += 1

        self.dx += self.dt * ddx  # compute the velocity
        # clamp velocity if needed
        if np.linalg.norm(self.dx) > self.max_dx:
            self.dx = self.max_dx * self.dx / np.linalg.norm(self.dx)

        return x


def clamp_kernel_1d(x, low_lim, high_lim, kernel_size):
    """
    A function to calculate the start and end indices
    of the kernel around the agent that is inside the grid
    i.e. clamp the kernel by the grid boundaries
    """
    start_kernel = low_lim
    start_grid = x - (kernel_size // 2)
    num_kernel = kernel_size
    # bound the agent to be inside the grid
    if x <= -(kernel_size // 2):
        x = -(kernel_size // 2) + 1
    elif x >= high_lim + (kernel_size // 2):
        x = high_lim + (kernel_size // 2) - 1

    # if agent kernel around the agent is outside the grid,
    # clamp the kernel by the grid boundaries
    if start_grid < low_lim:
        start_kernel = kernel_size // 2 - x - 1
        num_kernel = kernel_size - start_kernel - 1
        start_grid = low_lim
    elif start_grid + kernel_size >= high_lim:
        num_kernel -= x - (high_lim - num_kernel // 2 - 1)
    if num_kernel > low_lim:
        grid_indices = slice(start_grid, start_grid + num_kernel)

    return grid_indices, start_kernel, num_kernel


def agent_block(min_val, agent_radius):
    """
    A matrix representing the shape of an agent (e.g, RBF with Gaussian kernel). 
    min_val is the upper bound on the minimum value of the agent block.
    """
    eps = 1.0 / agent_radius  # shape parameter of the RBF
    l2_sqrd = (
        -np.log(min_val) / eps
    )  # squared maximum distance from the center of the agent block
    l2_sqrd_single = (
        l2_sqrd / param.nbVar
    )  # maximum squared distance on a single axis since sum of all axes equal to l2_sqrd
    l2_single = np.sqrt(l2_sqrd_single)  # maximum distance on a single axis
    # round to the nearest larger integer
    if l2_single.is_integer(): 
        l2_upper = int(l2_single)
    else:
        l2_upper = int(l2_single) + 1
    # agent block is symmetric about the center
    num_rows = l2_upper * 2 + 1
    num_cols = num_rows
    num_depths = num_rows
    block = np.zeros((num_depths, num_rows, num_cols))
    center = np.array([num_depths // 2, num_rows // 2, num_cols // 2])
    for k in range(num_depths):
        for i in range(num_rows):
            for j in range(num_cols):
                block[k, i, j] = rbf(np.array([k, i, j]), center, eps)
    return block


def offset(mat, i, j, k):
    """
    offset a 3D matrix by i, j, k
    """
    depths, rows, cols = mat.shape
    depths = depths - 2
    rows = rows - 2
    cols = cols - 2
    return mat[1 + i : 1 + i + depths, 1 + j : 1 + j + rows, 1 + k : 1 + k + cols]


def border_interpolate(x, length, border_type):
    """
    Helper function to interpolate border values based on the border type
    (gives the functionality of cv2.borderInterpolate function)
    """
    if border_type == "reflect101":
        if x < 0:
            return -x
        elif x >= length:
            return 2 * length - x - 2
    return x


def trilinear_interpolation(grid, pos):
    """
    Linear interpolating function on a 3-D grid
    """
    x, y, z = pos.astype(int)
    # find the nearest integers by minding the borders
    x0 = border_interpolate(x, grid.shape[1], "reflect101")
    x1 = border_interpolate(x + 1, grid.shape[1], "reflect101")
    y0 = border_interpolate(y, grid.shape[0], "reflect101")
    y1 = border_interpolate(y + 1, grid.shape[0], "reflect101")
    z0 = border_interpolate(z, grid.shape[0], "reflect101")
    z1 = border_interpolate(z + 1, grid.shape[0], "reflect101")
    # Distance from lower integers
    xd = pos[0] - x0
    yd = pos[1] - y0
    zd = pos[2] - z0
    # Interpolate on x-axis
    c00 = grid[z0, y0, x0] * (1 - xd) + grid[z0, y0, x1] * xd
    c01 = grid[z1, y0, x0] * (1 - xd) + grid[z1, y0, x1] * xd
    c10 = grid[z0, y1, x0] * (1 - xd) + grid[z0, y1, x1] * xd
    c11 = grid[z1, y1, x0] * (1 - xd) + grid[z1, y1, x1] * xd
    # Interpolate on y-axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    # Interpolate on z-axis
    c = c0 * (1 - zd) + c1 * zd
    return c


def discretize_gmm(param):
    alpha = np.ones(param.nbGaussian) / param.nbGaussian # mixing coeffs. Priors
    w_hat = fourier(alpha)

    # Fourier basis functions (for a discretized map)
    xm1d = np.linspace(param.xlim[0], param.xlim[1], param.nbRes)  # Spatial range
    xm_yy, xm_zz, xm_xx = np.meshgrid(xm1d, xm1d, xm1d)

    # Range
    KX = np.ndarray((param.nbVar, param.nbFct**param.nbVar))

    for n in range(param.nbVar):
        KX[n,:] = np.tile(np.repeat(param.rg, param.nbFct**n), param.nbFct**(param.nbVar-n-1))

    # Mind the flatten() !!!
    ang1 = (
        KX[0, :][:, np.newaxis]
        @ xm_xx.flatten()[:, np.newaxis].T
        * param.omega
    )
    ang2 = (
        KX[1, :][:, np.newaxis]
        @ xm_yy.flatten()[:, np.newaxis].T
        * param.omega
    )
    ang3 = (
        KX[2, :][:, np.newaxis]
        @ xm_zz.flatten()[:, np.newaxis].T
        * param.omega
    )
    phim = np.cos(ang1) * np.cos(ang2) * np.cos(ang3) * 2 ** (param.nbVar)
    # Some weird +1, -1 due to 0 index !!!
    yy, zz, xx = np.meshgrid(np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(param.nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1] * hk[zz.flatten() - 1]
    phim = phim * np.tile(HK, (param.nbRes**param.nbVar, 1)).T

    # Desired spatial distribution
    g = w_hat.T @ phim
    return g


def rbf(mean, x, eps):
    """
    Radial basis function w/ Gaussian Kernel
    """
    d = x - mean  # radial distance
    l2_norm_squared = np.dot(d, d)
    # eps is the shape parameter that can be interpreted as the inverse of the radius
    return np.exp(-eps * l2_norm_squared)


def normalize_mat(mat):
    return mat / (np.sum(mat) + 1e-10)


def calculate_gradient(x, agent, gradient_x, gradient_y, gradient_z):
    """
    Calculate movement direction of the agent by considering the gradient
    of the temperature field near the agent
    """
    # find agent pos on the grid as integer indices
    adjusted_position = x / param.dx
    # note x axis corresponds to col and y axis corresponds to row
    col, row, depth = adjusted_position.astype(int)

    gradient = np.zeros(3)
    # if agent is inside the grid, interpolate the gradient for agent position
    if row > 0 and row < param.height - 1 and col > 0 and col < param.width - 1 and depth > 0 and depth < param.depth - 1:
        gradient[0] = trilinear_interpolation(gradient_x, adjusted_position)
        gradient[1] = trilinear_interpolation(gradient_y, adjusted_position)
        gradient[2] = trilinear_interpolation(gradient_z, adjusted_position)

    # if kernel around the agent is outside the grid,
    # use the gradient to direct the agent inside the grid
    boundary_gradient = 0.1 #2  # 0.1
    pad = 0 #param.kernel_size - 1
    if row <= pad:
        gradient[1] = boundary_gradient
    elif row >= param.height - 1 - pad:
        gradient[1] = -boundary_gradient

    if col <= pad:
        gradient[0] = boundary_gradient
    elif col >= param.width - pad:
        gradient[0] = -boundary_gradient

    if depth <= pad:
        gradient[2] = boundary_gradient
    elif depth >= param.depth - pad:
        gradient[2] = -boundary_gradient

    return gradient #* param.dx


# Reset function
# ===============================
def reset(reset_state=True):
    global agent, goal_density, coverage_density, heat, coverage_block, param, trajectory

    # Retrieve the initial state defined by the user
    param.x0 = np.array(param.x0)
    if (len(param.x0.shape) != 1) or (param.x0.shape[0] != 7):
        print("Error: 'param.x0' must be a vector of size 7")
        return

    robot.jointPositions = param.x0

    # Retrieve the number of gaussians defined by the user, and create/delete existing ones as needed
    param.nbGaussian = max(int(param.nbGaussian), 1)
    update_gaussians(param)

    # Initialize the agent
    agent = SecondOrderAgent(param.nbVar, max_dx=param.max_dx, max_ddx=param.max_ddx)

    # Initialize heat equation related variables
    param.rg = np.arange(0, param.nbFct, dtype=float)
    param.kk = np.ndarray((param.nbVar, param.nbFct**param.nbVar))

    for n in range(param.nbVar):
        param.kk[n,:] = np.tile(np.repeat(param.rg, param.nbFct**n), param.nbFct**(param.nbVar-n-1))

    param.kk *= param.omega

    param.op = hadamard_matrix(2**(param.nbVar-1))
    param.op = np.array(param.op[:3,:])

    param.alpha = np.array([1, 1, 1]) * param.diffusion

    g = discretize_gmm(param)
    G = np.reshape(g, [param.nbRes, param.nbRes, param.nbRes])
    G = np.abs(G)    # there is no negative heat

    param.depth, param.height, param.width = G.shape

    param.area = param.dx * param.width * param.dx * param.height * param.dx * param.depth

    goal_density = normalize_mat(G)

    coverage_density = np.zeros((param.depth, param.height, param.width))
    heat = np.array(goal_density)

    max_diffusion = np.max(param.alpha)
    param.dt = min(
        1.0, (param.dx * param.dx * param.dx) / (8.0 * max_diffusion)
    )  # for the stability of implicit integration of Heat Equation
    coverage_block = agent_block(param.min_kernel_val, param.agent_radius)
    param.kernel_size = coverage_block.shape[0]

    # Other initializations
    trajectory = np.ndarray((1, 3))
    trajectory[0,:] = robot.fkin(robot.jointPositions)[:3]

    reset_rendering(param)


# Update function
# ===============================
def update():
    global agent, goal_density, coverage_density, heat, coverage_block, param, trajectory

    # Compute the command
    try:
        u, coverage_density, heat = controlCommand(
            robot.jointPositions,
            agent,
            goal_density,
            coverage_density,
            heat,
            coverage_block,
            param
        )
    except Exception as e:
        displayError(e)
        u = np.zeros(param.x0.shape)

    # Apply the command to the robot
    robot.control = robot.control + u * param.dt

    # Update the list of points used to draw the trajectory
    ee = robot.fkin(robot.jointPositions)
    if (trajectory.shape[0] < 3) or (np.linalg.norm(trajectory[-2,:] - ee[:3]) >= 0.05):
        trajectory = np.append(trajectory, [ee[:3]], axis=0)
    else:
        trajectory[-1,:] = ee[:3]
