
.. image:: images/MemoCellLogo.svg
   :width: 300px

.. image:: https://img.shields.io/pypi/v/memo_py.svg
    :target: https://pypi.python.org/pypi/memo_py
    :alt: Latest PyPI version

.. image:: https://travis-ci.org/borntyping/cookiecutter-pypackage-minimal.png
   :target: https://travis-ci.org/borntyping/cookiecutter-pypackage-minimal
   :alt: Latest Travis CI build status


An exact modelling framework for stochastic processes with and without **memo**\ ry in **Py**\ thon.

TODO:
Package name ideas:

*  Memopy
*  Phasepy
*  memoprocess (MemoProcess.jl)
*  memoryprocess (MemoryProcess.jl)
*  phaseprocess (PhaseProcess.jl)
*  ginkgo (Ginkgo.jl)

Consideration:

*  maybe find a name that would also fit to potential Julia package

Style guides:

*  Python packages should also have short, all-lowercase names; the use of underscores is discouraged
*  Julia: `see here <https://github.com/JuliaPraxis/Naming/blob/master/guides/PackagesAndModules.md>`_


Getting Started
---------------

Installation
^^^^^^^^^^^^

Memopy requires an installation of a recent Python version; Python can be
installed via `Anaconda <https://docs.anaconda.com/anaconda/install/>`_.

Make sure to have ``graphviz`` and ``pygraphviz`` installed before installing ``memo_py``; for
example by executing the following in the terminal::

   conda install graphviz
   conda install pygraphviz

Then ``memo_py`` can be installed by running::

   pip install memo_py

Other dependencies should be installed automatically during the ``memo_py`` installation.


First Example
^^^^^^^^^^^^^


Documentation
-------------

Documentation can be found here [TODO link to read the docs].

License
-------

This package can be used under the MIT License (MIT), see LICENSE file.

Authors
-------

Memopy was written and developed by `Maurice Langhinrichs <m.langhinrichs@icloud.com>`_ and `Lisa Buchauer <lisa.buchauer@posteo.de>`_ `@TSB <https://www.dkfz.de/en/modellierung-biologischer-systeme/>`_.

Citation
--------

The release paper of Memopy can be found here ``[TODO add link]``.

Please cite this publication as

``TODO add citation``
