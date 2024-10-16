import numpy as np


if 'gaussian_params' not in globals():
    gaussian_params = lambda: None # Lazy way to define an empty class in python

    # Gaussian centers (one per column)
    gaussian_params.Mu = np.array([
        [0.8, 0.8],
        [0.6, 0.5],
        [0.6, 0.5],
    ])

    # Gaussian covariances
    gaussian_params.Sigma = np.array([
        [
            [0.004, 0.003],
            [-0.03, -0.002],
            [0.003, 0.02],
        ],
        [
            [-0.03, -0.002],
            [0.301, 0.003],
            [-0.03, -0.02],
        ],
        [
            [0.003, 0.02],
            [-0.03, -0.02],
            [0.004, 0.201],
        ],
    ])


def hadamard_matrix(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    h = np.empty((n, n), dtype=int)
    h[:half_size,:half_size] = h_half
    h[half_size:,:half_size] = h_half
    h[:half_size:,half_size:] = h_half
    h[half_size:,half_size:] = -h_half
    return h


def fourier(alpha):
    ## Compute Fourier series coefficients w_hat of desired spatial distribution
    w_hat = np.zeros(param.nbFct**param.nbVar)
    for j in range(param.nbGaussian):
        for n in range(param.op.shape[1]):
            MuTmp = np.diag(param.op[:,n]) @ gaussian_params.Mu[:,j]
            SigmaTmp = np.diag(param.op[:,n]) @ gaussian_params.Sigma[:,:,j] @ np.diag(param.op[:,n]).T
            cos_term = np.cos(param.kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-.5 * param.kk.T @ SigmaTmp @ param.kk))
            w_hat = w_hat + alpha[j] * cos_term * exp_term
    return w_hat / (param.L**param.nbVar) / (param.op.shape[1])


def update_gaussians(param):
    if param.nbGaussian <= gaussian_params.Mu.shape[1]:
        return

    nbNewGaussian = param.nbGaussian - gaussian_params.Mu.shape[1]

    Mu = np.random.rand(param.nbVar, nbNewGaussian) * 0.8 + 0.1
    Mu[0,:] = np.random.rand(1, nbNewGaussian) * 0.2 + 0.7

    angles1 = np.random.rand(nbNewGaussian) * np.pi
    angles2 = np.random.rand(nbNewGaussian) * np.pi
    Sigma_vectors = np.ndarray((nbNewGaussian, param.nbVar))
    Sigma_vectors[:, 0] = np.cos(angles1)
    Sigma_vectors[:, 1] = np.sin(angles1)
    Sigma_vectors[:, 2] = np.sin(angles2)

    Sigma_scales = np.random.rand(nbNewGaussian) * 0.04 + 0.001
    Sigma_regularizations = np.random.rand(nbNewGaussian) * 1e-2 + 1e-4

    Sigma = np.zeros((param.nbVar, param.nbVar, nbNewGaussian))
    for i in range(nbNewGaussian):
        Sigma[:,:,i] = np.outer(Sigma_vectors[i,:], Sigma_vectors[i,:]) * Sigma_scales[i] + np.eye(param.nbVar) * Sigma_regularizations[i]

    gaussian_params.Mu = np.append(gaussian_params.Mu, Mu, axis=1)
    gaussian_params.Sigma = np.append(gaussian_params.Sigma, Sigma, axis=2)


# Forward kinematics function (allows to not care about 'robot' in the user code)
def fkin(x):
    return robot.fkin(x)

# Jacobian function (allows to not care about 'robot' in the user code)
def Jkin(x):
    return robot.Jkin(x)
