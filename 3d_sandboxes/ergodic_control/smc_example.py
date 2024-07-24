# Initial robot state
param.x0 = [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 1.0]

# Number of gaussians
param.nbGaussian = 2


def ergodicControl(x, t, wt, param):
    # Depends on the current position only here, outputs: dphi, phix, phiy, phiz
    ang = x[:, np.newaxis] * param.rg * param.omega
    phi1 = np.cos(ang)
    dphi1 = -np.sin(ang) * np.tile(param.rg, (param.nbVar, 1)) * param.omega
    phix = phi1[0, param.xx-1].flatten()
    phiy = phi1[1, param.yy-1].flatten()
    phiz = phi1[2, param.zz-1].flatten()
    dphix = dphi1[0, param.xx-1].flatten()
    dphiy = dphi1[1, param.yy-1].flatten()
    dphiz = dphi1[2, param.zz-1].flatten()
    dphi = np.vstack([[dphix * phiy * phiz], [phix * dphiy * phiz], [phix * phiy * dphiz]])

    # Depends on wt, wt starts with zeros, then updates
    wt = wt + (phix * phiy * phiz).T / (param.L**param.nbVar)

    # Controller with constrained velocity norm
    u = -dphi @ (param.Lambda * (wt/(t+1) - param.w_hat))
    u = u * param.u_max / (np.linalg.norm(u) + 0.1) # Velocity command

    # Update of the position
    x = x + u * param.dt

    return x, wt


def controlCommand(x, t, wt, param):
    J = Jkin(x)
    f = fkin(x)

    # Primary task: ergodic control
    e, wt = ergodicControl(f[:3], t, wt, param)
    u = np.linalg.pinv(J[:3,:]) @ (e - f[:3])

    # Secondary task: preferred pose maintenance
    N = np.eye(7) - np.linalg.pinv(J[:3,:]) @ J[:3,:] # Nullspace projection matrix
    xh = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 1])
    u = u + N @ (xh - x)

    return u, wt
