import numpy as np
from js import Path2D


## Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.nbFct = 10     # Number of basis functions along x and y
param.nbVar = 2      # Dimension of the space
param.nbGaussian = 2 # Number of Gaussians to represent the spatial distribution
param.xlim = [0, 1]  # Domain limit

param.x0 = np.array([[.2, .3], [.1, .8]]) # Initial points
param.nbAgents = 2
param.diffusion = 1  # increases global behavior
param.source_strength = 1    # increases local behavior
param.max_dx = 1 # maximum velocity of the agent
param.max_ddx = 0.5 # maximum acceleration of the agent
param.dx = 1
param.nbRes = 100   # resolution of discretization
param.min_kernel_val = 1e-8  # upper bound on the minimum value of the kernel
param.agent_radius = 10  # changes the effect of the agent on the coverage

# Timesteps for integrating diffusion (higher values lead to more global exploration, [1,100])
param.nb_diffusion_timesteps = 100  

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(0),xlim(1)]
param.omega = 2 * np.pi / param.L # omega


## Variables
# ===============================
agents = None
goal_density = None
coverage_density = None
heat = None
coverage_block = None

r_x = None


# HEDAC-related functions
# ===============================
class SecondOrderAgent:
    """
    A point mass agent with second order dynamics.
    """
    def __init__(
        self,
        x,
        max_dx=1,
        max_ddx=0.2,
    ):
        self.x = np.array(x)  # position
        # determine which dimension we are in from given position
        self.nbVarX = len(x)
        self.dx = np.zeros(self.nbVarX)  # velocity

        self.t = 0  # time
        self.dt = 1  # time step

        self.max_dx = max_dx
        self.max_ddx = max_ddx

    def update(self, gradient):
        """
        set the acceleration of the agent to clamped gradient
        compute the position at t+1 based on clamped acceleration
        and velocity
        """
        ddx = gradient # we use gradient of the potential field as acceleration
        # clamp acceleration if needed
        if np.linalg.norm(ddx) > self.max_ddx:
            ddx = self.max_ddx * ddx / np.linalg.norm(ddx)

        self.x = self.x + self.dt * self.dx + 0.5 * self.dt * self.dt * ddx
        self.t += 1

        self.dx += self.dt * ddx  # compute the velocity
        # clamp velocity if needed
        if np.linalg.norm(self.dx) > self.max_dx:
            self.dx = self.max_dx * self.dx / np.linalg.norm(self.dx)


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
    block = np.zeros((num_rows, num_cols))
    center = np.array([num_rows // 2, num_cols // 2])
    for i in range(num_rows):
        for j in range(num_cols):
            block[i, j] = rbf(np.array([j, i]), center, eps)
    return block


def offset(mat, i, j):
    """
    offset a 2D matrix by i, j
    """
    rows, cols = mat.shape
    rows = rows - 2
    cols = cols - 2
    return mat[1 + i : 1 + i + rows, 1 + j : 1 + j + cols]


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


def bilinear_interpolation(grid, pos):
    """
    Linear interpolating function on a 2-D grid
    """
    x, y = pos.astype(int)
    # find the nearest integers by minding the borders
    x0 = border_interpolate(x, grid.shape[1], "reflect101")
    x1 = border_interpolate(x + 1, grid.shape[1], "reflect101")
    y0 = border_interpolate(y, grid.shape[0], "reflect101")
    y1 = border_interpolate(y + 1, grid.shape[0], "reflect101")
    # Distance from lower integers
    xd = pos[0] - x0
    yd = pos[1] - y0
    # Interpolate on x-axis
    c01 = grid[y0, x0] * (1 - xd) + grid[y0, x1] * xd
    c11 = grid[y1, x0] * (1 - xd) + grid[y1, x1] * xd
    # Interpolate on y-axis
    c = c01 * (1 - yd) + c11 * yd
    return c


def discretize_gmm(param, KX):
    w_hat = fourier(param.Alpha)

    # Fourier basis functions (for a discretized map)
    xm1d = np.linspace(param.xlim[0], param.xlim[1], param.nbRes)    # Spatial range
    xm = np.zeros((param.nbVar, param.nbRes, param.nbRes))
    xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
    # Mind the flatten() !!!
    ang1 = (
        KX[0, :, :].flatten().T[:, np.newaxis]
        @ xm[0, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    ang2 = (
        KX[1, :, :].flatten().T[:, np.newaxis]
        @ xm[1, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    phim = np.cos(ang1) * np.cos(ang2) * 2 ** (param.nbVar)
    # Some weird +1, -1 due to 0 index !!!
    xx, yy = np.meshgrid(np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(param.nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
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


def calculate_gradient(agent, gradient_x, gradient_y):
    """
    Calculate movement direction of the agent by considering the gradient
    of the temperature field near the agent
    """
    # find agent pos on the grid as integer indices
    adjusted_position = agent.x / param.dx
    # note x axis corresponds to col and y axis corresponds to row
    col, row = adjusted_position.astype(int)

    gradient = np.zeros(2)
    # if agent is inside the grid, interpolate the gradient for agent position
    if row > 0 and row < param.height - 1 and col > 0 and col < param.width - 1:
        gradient[0] = bilinear_interpolation(gradient_x, adjusted_position)
        gradient[1] = bilinear_interpolation(gradient_y, adjusted_position)

    # if kernel around the agent is outside the grid,
    # use the gradient to direct the agent inside the grid
    boundary_gradient = 2  # 0.1
    pad = 0 #param.kernel_size - 1
    if row <= pad:
        gradient[1] = boundary_gradient
    elif row >= param.height - 1 - pad:
        gradient[1] = -boundary_gradient

    if col <= pad:
        gradient[0] = boundary_gradient
    elif col >= param.width - pad:
        gradient[0] = -boundary_gradient

    return gradient


# Reset function
# ===============================
def reset(reset_state=True):
    global agents, goal_density, coverage_density, heat, coverage_block, param, controls, paths, r_x

    # Retrieve the initial positions defined by the user
    param.x0 = np.array(param.x0)
    if (len(param.x0.shape) != 2) or (param.x0.shape[1] != 2):
        print("Error: 'param.x0' must be a Nx2 matrix, with 'N' the number of agents")
        return

    param.x0 = np.clip(param.x0, 0.01, 0.99) # x0 should be within [0,1]
    param.nbAgents = param.x0.shape[0]

    # Retrieve the number of gaussians defined by the user, and create/delete existing ones as needed
    param.nbGaussian = max(int(param.nbGaussian), 1)
    update_gaussians(param)

    # Initialize agents
    agents = []
    for i in range(param.nbAgents):
        agent = SecondOrderAgent(x=param.x0[i, :] * param.nbRes, max_dx=param.max_dx, max_ddx=param.max_ddx)
        agents.append(agent)

    # Initialize heat equation related fields
    param.alpha = np.array([1, 1]) * param.diffusion
    param.Alpha = (
        np.ones(param.nbGaussian) / param.nbGaussian
    )

    # Compute the desired spatial distribution
    param.rg = np.arange(0, param.nbFct, dtype=float)
    KX = np.zeros((param.nbVar, param.nbFct, param.nbFct))
    KX[0,:,:], KX[1,:,:] = np.meshgrid(param.rg, param.rg)

    param.op = hadamard_matrix(2**(param.nbVar-1))
    param.op = np.array(param.op)
    param.kk = KX.reshape(param.nbVar, param.nbFct**2) * param.omega

    g = discretize_gmm(param, KX)
    G = np.reshape(g, [param.nbRes, param.nbRes])
    G = np.abs(G)    # there is no negative heat

    param.height, param.width = G.shape

    param.area = param.dx * param.width * param.dx * param.height

    goal_density = normalize_mat(G)

    coverage_density = np.zeros((param.height, param.width))
    heat = np.array(goal_density)

    max_diffusion = np.max(param.alpha)
    param.dt = min(
        1.0, (param.dx * param.dx) / (4.0 * max_diffusion)
    )  # for the stability of implicit integration of Heat Equation
    coverage_block = agent_block(param.min_kernel_val, param.agent_radius)
    param.kernel_size = coverage_block.shape[0]

    # Other initializations
    r_x = np.array((0, 2))

    controls = create_gaussian_controls(param)
    paths = [ Path2D.new() for n in range(param.nbAgents) ]


# Update function
# ===============================
def update():
    global agents, goal_density, coverage_density, heat, coverage_block, param, paths, r_x

    x_prev = [ agent.x.copy() / param.nbRes for agent in agents ]

    coverage_density, heat = control(agents, goal_density, coverage_density, heat, coverage_block, param)

    x = [ agent.x.copy() / param.nbRes for agent in agents ]

    for path, prev, pos in zip(paths, x_prev, x):
        path.moveTo(prev[0], prev[1])
        path.lineTo(pos[0], pos[1])

    r_x = np.vstack((r_x, x))


# Rendering function
# ===============================
def draw_histograms():
    pass
