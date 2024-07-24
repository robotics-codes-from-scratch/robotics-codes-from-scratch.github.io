def initialState():
    """The initial state, customize it to your liking"""

    # Initial position
    x0 = 0.1

    # Gaussian centers (as many as you want, one per row)
    Mu = np.array([
        0.7,
    ])

    # Gaussian variances (one per row)
    Sigma = np.array([
        0.01,
    ])

    return (x0, Mu, Sigma)


def controlCommand(x, t, wt, param):
    # Fourier basis functions and derivatives for each dimension
    # (only cosine part on [0,L/2] is computed since the signal
    # is even and real by construction)
    phi = np.cos(x * param.kk)

    # Gradient of basis functions
    dphi = -np.sin(x * param.kk) * param.kk

    # Depends on wt, wt starts with zeros, then updates
    wt = wt + phi / param.L

    # Controller with constrained velocity norm
    u = -dphi.T @ (param.Lambda * (wt / (t + 1) - param.w_hat))
    u = u * param.u_max / (np.linalg.norm(u) + 1e-2)  # Velocity command

    return u, wt
