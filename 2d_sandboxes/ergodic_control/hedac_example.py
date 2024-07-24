# Initial position of the the agents (as many as you want, one per row)
param.x0 = np.array([
    [.2, .3],
])

# Number of Gaussians
param.nbGaussian = 2

# Timesteps for integrating diffusion (higher values lead to more global exploration, [1,100])
param.nb_diffusion_timesteps = 100  


def controlCommand(agents, goal_density, coverage_density, heat, coverage_block, param):
    # cooling of all the agents for a single timestep
    # this is used for collision avoidance bw/ agents
    for agent in agents:
        # find agent pos on the grid as integer indices
        p = agent.x
        adjusted_position = p / param.dx
        col, row = adjusted_position.astype(int)

        # each agent has a kernel around it,
        # clamp the kernel by the grid boundaries
        row_indices, row_start_kernel, num_kernel_rows = clamp_kernel_1d(
            row, 0, param.height, param.kernel_size
        )
        col_indices, col_start_kernel, num_kernel_cols = clamp_kernel_1d(
            col, 0, param.width, param.kernel_size
        )

        # add the kernel to the coverage density
        # effect of the agent on the coverage density
        coverage_density[row_indices, col_indices] += coverage_block[
            row_start_kernel : row_start_kernel + num_kernel_rows,
            col_start_kernel : col_start_kernel + num_kernel_cols,
        ]

    coverage = normalize_mat(coverage_density)

    # this is the part we introduce exploration problem to the Heat Equation
    diff = goal_density - coverage
    source = np.maximum(diff, 0) ** 2
    source = normalize_mat(source) * param.area

    # 2-D heat equation (Partial Differential Equation)
    # In 2-D we perform this second-order central for x and y.
    # Note that, delta_x = delta_y = h since we have a uniform grid.
    # Accordingly we have -4.0 of the center element.

    # At boundary we have Neumann boundary conditions which assumes
    # that the derivative is zero at the boundary. This is equivalent
    # to having a zero flux boundary condition or perfect insulation.
    for i in range(param.nb_diffusion_timesteps):
        heat[1:-1, 1:-1] = param.dt * (
            (
                + param.alpha[0] * offset(heat, 1, 0)
                + param.alpha[0] * offset(heat, -1, 0)
                + param.alpha[1] * offset(heat, 0, 1)
                + param.alpha[1] * offset(heat, 0, -1)
                - 4.0 * offset(heat, 0, 0)
            )
            / (param.dx * param.dx)
            + param.source_strength * offset(source, 0, 0)
        ) + offset(heat, 0, 0)

    # Calculate the first derivatives mind the order x and y
    gradient_y, gradient_x = np.gradient(heat, 1, 1)

    for agent in agents:
        grad = calculate_gradient(
            agent,
            gradient_x,
            gradient_y,
        )
        agent.update(grad)

    return coverage_density, heat
