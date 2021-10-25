from setuptools import setup, find_packages


PACKAGENAME = "ellipsoidal_nfw"
VERSION = "0.0.1"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Ellipsoidal NFW halo profiles",
    long_description="Ellipsoidal NFW halo profiles",
    install_requires=["numpy", "scipy"],
    packages=find_packages(),
    url="https://github.com/aphearin/ellipsoidal_nfw",
)
