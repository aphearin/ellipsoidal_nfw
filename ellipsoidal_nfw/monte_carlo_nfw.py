"""Module generates a random 3d positions according to a triaxial NFW profile."""
import numpy as np
from scipy import special


def random_nfw_ellipsoid(conc, a=1, b=1, c=1, seed=None):
    """Generate random points within an NFW ellipsoid with unit radius.

    Parameters
    ----------
    conc : ndarray
        Array of concentrations of shape (n, )

    a : float or ndarray of shape (n, ), optional
        Length of the x-axis. Default is 1 for a unit sphere.

    b : float or ndarray of shape (n, ), optional
        Length of the y-axis. Default is 1 for a unit sphere.

    c : float or ndarray of shape (n, ), optional
        Length of the z-axis. Default is 1 for a unit sphere.

    seed : int, optional
        Random number seed

    Returns
    -------
    x, y, z : ndarrays of shape (n, )

    Notes
    -----
    The halotools.utils.rotate_vector_collection may be useful for rotating
    the returned ellipsoid.

    Each individual point is permitted to have its own concentration.
    For example, to generate 5000 points within a single of a halo with concentration 5,
    use conc = np.zeros(5000) + 5.

    """
    x, y, z = random_nfw_spherical_coords(conc, seed=seed)
    return a * x, b * y, c * z


def random_nfw_spherical_coords(conc, seed=None):
    """Generate random points within an NFW sphere with unit radius.

    Parameters
    ----------
    conc : ndarray
        Array of concentrations of shape (n, )

    seed : int, optional
        Random number seed

    Returns
    -------
    x, y, z : ndarrays of shape (n, )

    """
    conc = np.atleast_1d(conc)
    npts = conc.size

    rng = np.random.RandomState(seed)
    randoms = rng.uniform(0, 1, 3 * npts)

    r = random_nfw_radial_position(conc, seed=seed, randoms=randoms[:npts])
    x, y, z = _random_spherical_position(randoms[npts:])

    return r * x, r * y, r * z


def _random_spherical_position(u):
    """Generate random points on the surface of 1d sphere.

    Parameters
    ----------
    u : ndarray of shape (2*n, )
        Uniform random points in the range (0, 1)

    Returns
    -------
    x, y, z : ndarrays of shape (n, )

    """
    n = u.size
    nhalf = n // 2
    cos_t = 2 * u[:nhalf] - 1
    phi = 2 * np.pi * u[nhalf:]

    sin_t = np.sqrt((1.0 - cos_t * cos_t))

    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t

    return x, y, z


def random_nfw_radial_position(conc, seed=None, randoms=None):
    """Generate random radial positions according to an NFW profile.

    The returned positions are dimensionless, and so should be multiplied by the
    halo radius to generate halo-centric distances.

    Parameters
    ----------
    conc : ndarray
        Array of concentrations of shape (n, )

    seed : int, optional
        Random number seed

    randoms : ndarray, optional
        Array of shape (n, ) of the CDF to use for each value of concentration

    Returns
    -------
    r : ndarray
        Array of shape (n, ) storing r/Rhalo, so that 0 < x < 1.
    """
    conc = np.atleast_1d(conc)
    n = conc.size
    if randoms is None:
        rng = np.random.RandomState(seed)
        u = rng.rand(n)
    else:
        u = np.atleast_1d(randoms)
        assert u.size == n, "randoms must have the same size as conc"
        assert np.all(randoms >= 0), "randoms must be non-negative"
        assert np.all(randoms <= 1), "randoms cannot exceed unity"

    return _qnfw(u, conc)


def _pnfwunorm(q, conc):
    """ """
    y = q * conc
    return np.log(1.0 + y) - y / (1.0 + y)


def _qnfw(p, conc):
    """ """
    assert np.all(p >= 0), "randoms must be non-negative"
    assert np.all(p <= 1), "randoms cannot exceed unity"
    p *= _pnfwunorm(1, conc)
    return (-(1.0 / np.real(special.lambertw(-np.exp(-p - 1)))) - 1) / conc
