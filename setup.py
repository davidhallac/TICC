#from distutils.core import setup
from setuptools import setup

setup(
	name='TICC',
	version='0.1.6',
	description='Solver for Toeplitz Inverse Covariance-based Clustering (TICC)',
	url='https://github.com/davidhallac/ticc',
	download_url='https://github.com/davidhallac/TICC/blob/blackbox/tarFile/TICC-0.1.6.tar.gz',
	install_requires=[
          'numpy', 'scipy', 'matplotlib', 'pandas','scikit-learn'
      ],
	packages=['ticc']
	)
