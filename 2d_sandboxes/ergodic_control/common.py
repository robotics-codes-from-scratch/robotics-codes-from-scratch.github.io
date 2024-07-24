import numpy as np


if 'gaussians' not in globals():
    gaussians = lambda: None # Lazy way to define an empty class in python

    # Gaussian centers (one per column)
    gaussians.Mu = np.array([
        [0.3, 0.7],
        [0.7, 0.3],
    ])

    # Gaussian covariances
    gaussians.Sigma = np.array([
        [
            [0.01, 0.008],
            [0.003, 0.01],
        ],
        [
            [0.003, 0.01],
            [0.002, 0.023],
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
            MuTmp = np.diag(param.op[:,n]) @ gaussians.Mu[:,j]
            SigmaTmp = np.diag(param.op[:,n]) @ gaussians.Sigma[:,:,j] @ np.diag(param.op[:,n]).T
            cos_term = np.cos(param.kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-.5 * param.kk.T @ SigmaTmp @ param.kk))
            w_hat = w_hat + alpha[j] * cos_term * exp_term
    return w_hat / (param.L**param.nbVar) / (param.op.shape[1])


def update_gaussians(param):
    if param.nbGaussian <= gaussians.Mu.shape[1]:
        return

    nbNewGaussian = param.nbGaussian - gaussians.Mu.shape[1]

    Mu = np.random.rand(2, nbNewGaussian) * 0.8 + 0.1

    angles = np.random.rand(nbNewGaussian) * np.pi
    Sigma_vectors = np.ndarray((nbNewGaussian, 2))
    Sigma_vectors[:, 0] = np.cos(angles)
    Sigma_vectors[:, 1] = np.sin(angles)

    Sigma_scales = np.random.rand(nbNewGaussian) * 0.04 + 0.001
    Sigma_regularizations = np.random.rand(nbNewGaussian) * 1e-2 + 1e-4

    Sigma = np.zeros((param.nbVar, param.nbVar, nbNewGaussian))
    for i in range(nbNewGaussian):
        Sigma[:,:,i] = np.outer(Sigma_vectors[i,:], Sigma_vectors[i,:]) * Sigma_scales[i] + np.eye(param.nbVar) * Sigma_regularizations[i]

    gaussians.Mu = np.append(gaussians.Mu, Mu, axis=1)
    gaussians.Sigma = np.append(gaussians.Sigma, Sigma, axis=2)
