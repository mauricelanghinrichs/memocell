
import io
import os
import re

from setuptools import find_packages
from setuptools import setup

# inactive for the moment, make pypi description simple
# def read(filename):
#     filename = os.path.join(os.path.dirname(__file__), filename)
#     text_type = type(u"")
#     with io.open(filename, mode="r", encoding='utf-8') as fd:
#         return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="memocell",
    version="0.1.5",
    url="https://github.com/mauricelanghinrichs/memocell.git",
    license='MIT',

    author="Maurice Langhinrichs",
    author_email="m.langhinrichs@icloud.com",

    description="Bayesian inference of stochastic cellular processes with and without memory in Python.",

    # README is rst currently, switch to md / markdown for pypi:
    # long_description_content_type="text/markdown"
    long_description_content_type="text/x-rst",
    long_description="Please visit MemoCell on `GitHub <https://github.com/mauricelanghinrichs/memocell>`_.",

    include_package_data=True,
    packages=find_packages(exclude=('tests',)),

    # package dependendies
    install_requires=['numpy>=1.16.5', 'sympy>=1.3', 'networkx>=2.2',
                        'scipy>=1.0.0', 'corner>=2.0.1', 'dynesty>=0.9',
                        'matplotlib>=2.1.2', 'seaborn>=0.9.0', 'cycler>=0.10.0',
                        'setuptools>=38.5.1', 'numba>=0.43.1',
                        # 'graphviz>=0.10.1', 'pygraphviz>=1.5', # should be installed separately before
                        'pydot>=1.4.1', 'psutil>=5.0.0',
                        'IPython>=7.8.0', 'tqdm>=4.31.1',
                        'ipywidgets>=7.5.1'],

    # run pytest with "$ python setup.py pytest" in this directory
    test_suite='tests',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    python_requires='>=3.6',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
