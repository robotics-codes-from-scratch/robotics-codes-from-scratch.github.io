# Initial robot state
param.x0 = [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 1.0]

# Number of Gaussians
param.nbGaussian = 2

# Timesteps for integrating diffusion (higher values lead to more global exploration, [1,25])
param.nb_diffusion_timesteps = 25 


def ergodicControl(x, agent, goal_density, coverage_density, heat, coverage_block, param):
    # find agent pos on the grid as integer indices
    p = x * param.nbRes
    adjusted_position = p / param.dx
    col, row, depth = adjusted_position.astype(int)

    # each agent has a kernel around it,
    # clamp the kernel by the grid boundaries
    row_indices, row_start_kernel, num_kernel_rows = clamp_kernel_1d(
        row, 0, param.height, param.kernel_size
    )
    col_indices, col_start_kernel, num_kernel_cols = clamp_kernel_1d(
        col, 0, param.width, param.kernel_size
    )
    depth_indices, depth_start_kernel, num_kernel_depths = clamp_kernel_1d(
        depth, 0, param.depth, param.kernel_size
    )

    # add the kernel to the coverage density
    # effect of the agent on the coverage density
    coverage_density[depth_indices, row_indices, col_indices] += coverage_block[
        depth_start_kernel : depth_start_kernel + num_kernel_depths,
        row_start_kernel : row_start_kernel + num_kernel_rows,
        col_start_kernel : col_start_kernel + num_kernel_cols,
    ]

    coverage = normalize_mat(coverage_density)

    # this is the part we introduce exploration problem to the Heat Equation
    diff = goal_density - coverage
    source = np.maximum(diff, 0) ** 3
    source = normalize_mat(source) * param.area

    # 3-D heat equation (Partial Differential Equation)
    # In 3-D we perform this second-order central for x,y and z
    # Note that, delta_x = delta_y = delta_z = h since we have a uniform grid.
    # Accordingly we have -6.0 of the center element.
    
    # At boundary we have Neumann boundary conditions which assumes
    # that the derivative is zero at the boundary. This is equivalent
    # to having a zero flux boundary condition or perfect insulation.
    for i in range(param.nb_diffusion_timesteps):
        heat[1:-1, 1:-1, 1:-1] = param.dt * (
            (
                + param.alpha[2] * offset(heat, 1, 0, 0)
                + param.alpha[2] * offset(heat, -1, 0, 0)
                + param.alpha[1] * offset(heat, 0, 1, 0)
                + param.alpha[1] * offset(heat, 0, -1, 0)
                + param.alpha[0] * offset(heat, 0, 0, 1)
                + param.alpha[0] * offset(heat, 0, 0, -1)
                - 6.0 * offset(heat, 0, 0, 0)
            )
            / (param.dx * param.dx * param.dx)
            + param.source_strength * offset(source, 0, 0, 0)
        ) + offset(heat, 0, 0, 0)

    # Calculate the first derivatives (mind the order x, y and z)
    gradient_z, gradient_y, gradient_x = np.gradient(heat, 1, 1, 1)

    grad = calculate_gradient(
        p,
        agent,
        gradient_x,
        gradient_y,
        gradient_z,
    )
    p = agent.update(p, grad)

    return p / param.nbRes, coverage_density, heat


def controlCommand(x, agent, goal_density, coverage_density, heat, coverage_block, param):
    J = Jkin(x)
    f = fkin(x)

    # Primary task: ergodic control
    e, coverage_density, heat = ergodicControl(f[:3], agent, goal_density, coverage_density, heat, coverage_block, param)
    u = np.linalg.pinv(J[:3,:]) @ (e - f[:3])

    # Secondary task: preferred pose maintenance
    N = np.eye(7) - np.linalg.pinv(J[:3,:]) @ J[:3,:] # Nullspace projection matrix
    xh = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 1])
    u = u + N @ (xh - x)

    return u, coverage_density, heat
