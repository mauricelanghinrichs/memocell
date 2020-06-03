
import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="memo_py", # memo<rho>y
    version="0.1.0",
    url="https://github.com/mauricelanghinrichs/memo_py.git",
    license='MIT',

    author="Maurice Langhinrichs",
    author_email="m.langhinrichs@icloud.com",

    description="An exact modelling framework for stochastic processes with and without memory in Python.",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    # package dependendies
    install_requires=['numpy>=1.14.1', 'sympy>=1.1.1', 'networkx>=2.2',
                        'scipy>=1.0.0', 'corner>=2.0.1', 'dynesty>=0.9',
                        'matplotlib>=2.1.2', 'cycler>=0.10.0', 'setuptools>=38.5.1',
                        'numba>=0.43.1', 'graphviz>=0.10.1', 'pygraphviz>=1.5',
                        'tqdm>=4.31.1'],

    # run pytest with "$ python setup.py pytest" in this directory
    test_suite='tests',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
