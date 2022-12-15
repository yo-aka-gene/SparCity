# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['sparcity',
 'sparcity.core',
 'sparcity.dataset',
 'sparcity.dataset.data',
 'sparcity.debug_utils',
 'sparcity.dev',
 'sparcity.gaussian_process',
 'sparcity.metrics',
 'sparcity.random',
 'sparcity.sampler']

package_data = \
{'': ['*']}

install_requires = \
['gpy>=1.10.0,<2.0.0',
 'gpytorch>=1.9.0,<2.0.0',
 'numpy>=1.23.5,<2.0.0',
 'pandas>=1.5.1,<2.0.0',
 'scipy>=1.9.3,<2.0.0',
 'torch==1.11.0',
 'tqdm>=4.64.1,<5.0.0']

setup_kwargs = {
    'name': 'sparcity',
    'version': '0.1.0',
    'description': 'Sparse estimator for geographical information',
    'long_description': 'Sparse estimator for geographical information\n',
    'author': 'yo-aka-gene',
    'author_email': 'yujiokano@keio.jp',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}


setup(**setup_kwargs)
