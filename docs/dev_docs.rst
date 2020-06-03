
Developers' Documentation
=========================

Running Tests
^^^^^^^^^^^^^

Running tests requires the ``pytest`` package, available via ``pip`` or ``conda``.

Go into the Memopy directory and run::

   python setup.py pytest

Alternatively, go into the ``./tests/`` directory and run::

   py.test test_memo_py.py


Building Documentation
^^^^^^^^^^^^^^^^^^^^^^

Requirements are ``sphinx`` and the ``sphinx_rtd_theme``, available via ``pip``. Go into the ``./docs/`` directory and run::

   make html

A local html file can then be found at ``./docs/_build/html/index.html``.
