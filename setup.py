from setuptools import setup, find_packages
from corrcoeff import __author__, __version__, __url__

setup(name="corrcoeff",
      version = __version__,
      packages = find_packages(),
      description = "Correlation between CMB temperature and polarization",
      url = __url__,
      author = __author__,
      keywords = ["CMB", "correlation", "planck", "SO"],
      classifiers = ["Intended Audience :: Science/Research",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Operating System :: OS Independent",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3.7"],
      install_requires = ["camb", "cobaya"],
      entry_points = {
        "console_scripts": ["corrcoeff=corrcoeff.corrcoeff:main"],
      }
)
