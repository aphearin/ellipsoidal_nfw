"""
"""
from ..monte_carlo_nfw import random_nfw_ellipsoid
import numpy as np


def test1():
    npts = 2000
    conc = np.zeros(npts) + 5
    x, y, z = random_nfw_ellipsoid(conc)
    r = np.sqrt(x * x + y * y + z * z)
    assert r.shape == (npts,)
    assert np.all(r > 0)
    assert np.all(r < 1)
