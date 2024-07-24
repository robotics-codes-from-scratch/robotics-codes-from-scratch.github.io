# Number of gaussians
param.nbGaussian = 2


def controlCommand(x, t, wt, param):
    # Depends on the current position only here, outputs: dphi, phix, phiy
    ang = x[:, np.newaxis] * param.rg * param.omega
    phi1 = np.cos(ang)
    dphi1 = -np.sin(ang) * np.tile(param.rg, (param.nbVar, 1)) * param.omega
    phix = phi1[0, param.xx-1].flatten()
    phiy = phi1[1, param.yy-1].flatten()
    dphix = dphi1[0, param.xx-1].flatten()
    dphiy = dphi1[1, param.yy-1].flatten()
    dphi = np.vstack([[dphix * phiy], [phix * dphiy]])

    # Depends on wt, wt starts with zeros, then updates
    wt = wt + (phix * phiy).T / (param.L**param.nbVar)

    # Controller with constrained velocity norm
    u = -dphi @ (param.Lambda * (wt/(t+1) - param.w_hat))
    u = u * param.u_max / (np.linalg.norm(u) + 1e-1) # Velocity command

    return u, wt
