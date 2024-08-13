import jax.numpy as jnp


def rbf(obst, prediction_vector, tau):
    """RBF kernel
    Args:
        obst: input vector
        pret: prediction vector
        tau: scale

    Returns:
        kernel
    """
    Dt = obst - jnp.array([prediction_vector]).T
    return jnp.exp(-((Dt) ** 2) / 2 / (tau**2))


def matern32(obst, prediction_vector, tau):
    """Matern 3/2 kernel
    Args:
        obst: input vector
        pret: prediction vector
        tau: scale

    Returns:
        kernel
    """

    Dt = obst - jnp.array([prediction_vector]).T
    fac = jnp.sqrt(3.0) * jnp.abs(Dt) / tau
    return (1.0 + fac) * jnp.exp(-fac)


def gp2d(input_matrix, gpkernel, sigma, scale, pshape=None):
    """GP 2D for different size between input and prediction.

    Args:
        input_matrix: input 2D matrix
        gpkernel: GP kernel
        sigma: observational Gaussian noise std
        xscale: GP correlated length (hyperparameter) for (X,Y)
        kernel: GP kernel, rbf or matern32
        pshape: prediction matrix shape. If None, use the same shape as Dmat

    Returns:
        prediction 2D matrix
    """
    if pshape == None:
        pshape = jnp.shape(input_matrix)
    rat = jnp.array(pshape) / jnp.array(jnp.shape(input_matrix))
    Nx, Ny = jnp.shape(input_matrix)
    fNx = float(Nx)
    fNy = float(Ny)
    x = (jnp.arange(0.0, fNx)) * rat[0]
    y = (jnp.arange(0.0, fNy)) * rat[1]
    Nxp, Nyp = pshape
    fNxp = float(Nxp)
    fNyp = float(Nyp)
    xp = jnp.arange(0.0, fNxp)
    yp = jnp.arange(0.0, fNyp)
    Kx = gpkernel(x, x, scale[0])
    Ky = gpkernel(y, y, scale[1])
    kapx, Ux = jnp.linalg.eigh(Kx)
    kapy, Uy = jnp.linalg.eigh(Ky)
    invL = 1.0 / (jnp.outer(kapx, kapy) + sigma**2)
    P = invL * (jnp.dot(Ux.T, jnp.dot(input_matrix, Uy)))
    Kxp = gpkernel(x, xp, scale[0])
    Kyp = gpkernel(y, yp, scale[1])
    prediction_matrix = Kxp @ Ux @ P @ Uy.T @ Kyp.T
    return prediction_matrix
