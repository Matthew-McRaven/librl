from distutils.core import setup
from setuptools import setup, find_packages
import pkg_resources
install_reqs = []
install_reqs.extend(["more-itertools", "overrides", "future", "matplotlib", "scipy", "Pillow"])
install_reqs.extend(["networkx ~= 2.5", "torch ~= 1.6.0", "torchvision ~= 0.7.0"])
setup(name='librl',
      version='0.1.7',
      description='Library for various DRL and DL projects.',
      author='Matthew McRaven',
      author_email='mkm302@georgetown.edu',
      url='https://github.com/Matthew-McRaven/librl',
      install_reqs = install_reqs,
      packages=find_packages()
     )