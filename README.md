# gpkrax

A simple and fast differentiable [2D gaussian process fitting using Kronecker product](https://github.com/HajimeKawahara/gpkron/blob/main/documents/pdf/GP2D.pdf).

```sh
python setup.py install
```

## sample

```python
    from gpkrax.gp2d import gp2d, rbf, matern32
    np.random.seed(seed=1)

    Nx = 16; Ny = 32
    pshape=(64,128)

    xgrid = jnp.linspace(0, Nx, Nx)
    ygrid = jnp.linspace(0, Ny, Ny)
    sigma = 0.2
    Dmat = np.sin(xgrid[:, np.newaxis]/4) * np.sin(ygrid[np.newaxis, :]/4) + \
        np.random.randn(Nx, Ny)*sigma

    Dprer = gp2d(Dmat, rbf, sigma, (20., 20.), pshape=pshape)
    Dprem = gp2d(Dmat, matern32, sigma, (40., 40.), pshape=pshape)
```

Run tests/integration/gp2d_sample.py

![sample](https://github.com/user-attachments/assets/25df1a9f-40e1-4256-9740-8e7f06a5e0bb)
